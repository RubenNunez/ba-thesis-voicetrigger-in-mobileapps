import sounddevice as sd
import threading
import time
import argparse
import torchaudio
import torch
import numpy as np
from threading import Event

class Listener:

    def __init__(self, sample_rate=8000, record_seconds=10):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.audio_buffer = []

    def callback(self, indata, frames, time, status):
        self.audio_buffer.append(indata.copy())

    def listen(self):
        with sd.InputStream(callback=self.callback, samplerate=self.sample_rate, channels=1, blocksize=self.chunk, dtype='int16'):
            
            while True:
                time.sleep(0.01)
            #sd.sleep(int(self.record_seconds * 1000))

    def run(self):
        thread = threading.Thread(target=self.listen, daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening...\n")

class WakeupEngine:

    def __init__(self, model_file):
        self.listener = Listener(sample_rate=8000, record_seconds=10)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')
        self.featurizer = torchaudio.transforms.MFCC(
            sample_rate=8000, n_mfcc=40, melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40})

    def predict(self):
        if len(self.listener.audio_buffer) > 0:
            audio = np.concatenate(self.listener.audio_buffer)
            self.listener.audio_buffer.clear()
            with torch.no_grad():
                waveform = torch.Tensor(audio).reshape(1, -1)
                mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)
                out = self.model(mfcc)
                out_value = out.item()
                print(f"Model output: {out_value}")
                threshold = 0.5  # Adjust as necessary
                pred = 1.0 if out_value > threshold else 0.0
                return pred
                pred = torch.sigmoid(out)
                return pred.item()
        return 0  # No prediction if buffer is empty

    def inference_loop(self, action):
        while True:
            prob = self.predict()
            if prob > 0.01:
                line = f"\rWakeup Word Probability: {prob:.10f}"
                print(line)


            action(prob)
            time.sleep(0.05)

    def run(self, action):
        self.listener.run()
        thread = threading.Thread(target=self.inference_loop, args=(action,), daemon=True)
        thread.start()

class DemoAction:

    def __init__(self, sensitivity=10):
        self.detect_in_row = 0
        self.sensitivity = sensitivity

    def __call__(self, prediction):
        if prediction == 1:
            self.detect_in_row += 1
            if self.detect_in_row == self.sensitivity:
                self.detected()
                self.detect_in_row = 0
        else:
            self.detect_in_row = 0

    def detected(self):
        print("\nWakeword Detected!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the wakeword engine")
    parser.add_argument('--model_file', type=str, default=None, required=True,
                        help='optimized file to load. use optimize_graph.py')
    parser.add_argument('--sensitivity', type=int, default=10, required=False,
                        help='lower value is more sensitive to activations')

    args = parser.parse_args()
    wakeword_engine = WakeupEngine(args.model_file)
    action = DemoAction(sensitivity=args.sensitivity)

    wakeword_engine.run(action)
    threading.Event().wait()
