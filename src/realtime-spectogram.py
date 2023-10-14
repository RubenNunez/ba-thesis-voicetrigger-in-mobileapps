# Realtime Spectogram

import sys
import numpy as np
import pyaudio
from PyQt5 import QtWidgets
import pyqtgraph as pg

CHUNK = 1024
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1

class AudioStream(object):
    def __init__(self):
        # Initialisiere Audio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        # Initialisiere PyQtGraph
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsWindow(title="Realtime Spectrogram")
        self.specItem = self.win.addPlot(title="Spectrogram", labels={'left': 'Frequency', 'bottom': 'Time'})
        self.img = pg.ImageItem()
        self.specItem.addItem(self.img)
        self.specItem.setAspectLocked(False)
        self.img_array = np.zeros((1000, CHUNK//2+1))

    def start(self):
        timer = pg.QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(10)  # Aktualisierungsrate in Millisekunden
        self.app.exec_()

    def update(self):
        data = np.fromstring(self.stream.read(CHUNK, exception_on_overflow=False), np.int16)
        freq, time, Sxx = self.get_spectrogram(data)
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = 10 * np.log10(Sxx)
        self.img.setImage(self.img_array, autoLevels=True)

    def get_spectrogram(self, data):
        freqs = np.fft.rfftfreq(CHUNK, 1./RATE)
        t = np.arange(0, CHUNK, 1)
        Sxx = np.fft.rfft(data)  # FFT
        return freqs, t, np.abs(Sxx)

if __name__ == '__main__':
    audio_app = AudioStream()
    res = audio_app.start()
    sys.exit(res)

