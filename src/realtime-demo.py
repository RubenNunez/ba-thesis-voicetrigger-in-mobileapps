"""PyAudio Example: Record a few seconds of audio and save to a wave file."""

import wave
import sys

import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 5

#p = pyaudio.PyAudio()
#for i in range(p.get_device_count()):
#    info = p.get_device_info_by_index(i)
#    print(info['index'], info['name'])

with wave.open('output.wav', 'wb') as wf:
    p = pyaudio.PyAudio()
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=5)
    
    print('Recording...')
    for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
        wf.writeframes(stream.read(CHUNK))
    print('Done')
    
    stream.close()
    p.terminate()