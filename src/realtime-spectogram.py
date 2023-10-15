import sys
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import pyqtgraph as pg
import colorcet as cc
import matplotlib.pyplot as plt
import signal # For Ctrl+C handling

signal.signal(signal.SIGINT, signal.SIG_DFL)



# https://colorcet.holoviz.org/
# Default colormap
colormap = pg.colormap.get("CET-R1")  # 'CET-R1', 'CET-D1', 'magma', 'inferno', 'plasma',
default_lut = colormap.getLookupTable()

colors = plt.get_cmap(cc.m_fire)(np.linspace(0, 1, 256))
default_lut = (colors[:, :3] * 255).astype(np.uint8)



CHUNK = 1024
RATE = 44100
CHANNELS = 1

class AudioStream(object):

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.app.quit()

    def __init__(self):
        self.stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype=np.int16)
        self.stream.start()
        
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Realtime Spectrogram & Waveform")
        
        # Spektrogramm
        self.specItem = self.win.addPlot(title="Spectrogram", labels={'left': 'Frequency', 'bottom': 'Time (ms)'},)
        self.img = pg.ImageItem()
        self.specItem.addItem(self.img)
        self.specItem.setAspectLocked(False)
        self.img_array = np.zeros((1000, CHUNK//2+1))

        # Waves
        self.win.nextRow()
        self.waveItem = self.win.addPlot(title="Waveform", labels={'left': 'Amplitude', 'bottom': 'Samples'})
        
        # Colormap Spektrogramm
        self.img.setLookupTable(default_lut)

        # X-Axis Spektrogramm in (ms) instead of samples
        self.time_ticks = [(i, str(i*10)) for i in range(0, 1001, 100)]  # Jeder 100. Punkt wird gelabelt
        self.specItem.getAxis('bottom').setTicks([self.time_ticks])

        self.win.keyPressEvent = self.keyPressEvent

        self.win.show() 

    def start(self):
        timer = pg.QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(10)  # Aktualisierungsrate in Millisekunden
        self.app.exec_()

    def update(self):
        data, _ = self.stream.read(CHUNK)
        data = data.flatten()
        
        # Update the waveform
        self.waveItem.plot(data, clear=True, pen=(100,100,100))
        
        # Update the spectrogram
        __freqs, _time, Sxx = self.get_spectrogram(data)
        Sxx[Sxx == 0] = 1e-10  # avoid zero division
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = 10 * np.log10(Sxx)
        self.img.setImage(self.img_array, autoLevels=False, levels=[-80, 80])  # Colormap: -50dB bis 50dB (dynamic range)
                 

    def get_spectrogram(self, data):
        freqs = np.fft.rfftfreq(CHUNK, 1./RATE)
        t = np.arange(0, CHUNK, 1)
        Sxx = np.fft.rfft(data)  # FFT
        Sxx = np.abs(np.fft.rfft(data))**2
        return freqs, t, np.abs(Sxx)

if __name__ == '__main__':
    audio_app = AudioStream()
    res = audio_app.start()
    sys.exit(res)
