import sys
import wave
import sounddevice as sd
import numpy as np

def record_audio_to_wav_file(filename, record_seconds, samplerate=44100, channels=2, device=0):
    
    if sys.platform == 'darwin':
        channels = 1

    print('Recording...')
    recording = sd.rec(int(samplerate * record_seconds), 
                       samplerate=samplerate, 
                       channels=channels, 
                       device=device,
                       dtype='int16',)
    sd.wait()  
    print('Recording completed.')

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes for 'int16'
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())

if __name__ == '__main__':
    
    default_input_index = sd.default.device[0]
    default_input_name = sd.query_devices()[default_input_index]['name']

    print(f'Default input device index: {default_input_index}, Name: {default_input_name}')
    
    print('\n')

    print('Available input devices:')

    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(idx, device['name'])
    
    print('\n')
    print('Please select the input device index:')

    device_idx = int(input())
    record_audio_to_wav_file('output.wav', 10, device=device_idx)
