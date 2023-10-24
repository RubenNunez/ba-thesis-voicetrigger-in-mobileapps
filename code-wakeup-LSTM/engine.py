"""the interface to interact with wakeword model"""
import sounddevice as sd
import threading
import time
import argparse
import wave
import torchaudio
import torch
import numpy as np
from dataset import get_featurizer
from threading import Event


class Listener:

    def __init__(self, sample_rate=8000, record_seconds=2):
        self.chunk = 1024
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds

    def callback(self, indata, frames, time, status, queue):
        # Diese Funktion wird jedes Mal aufgerufen, wenn genug Audio-Daten verfügbar sind
        queue.append(indata.copy())

    def listen(self, queue):
        with sd.InputStream(callback=lambda indata, frames, time, status: self.callback(indata, frames, time, status, queue), samplerate=self.sample_rate, channels=1, blocksize=self.chunk) as stream:
            # Aufnahmezeit in Millisekunden
            sd.sleep(self.record_seconds * 1000)

    def run(self, queue):
        thread = threading.Thread(
            target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\nWake Word Engine is now listening... \n")


class WakeupEngine:

    def __init__(self, model_file):
        self.listener = Listener(sample_rate=8000, record_seconds=2)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  # run on cpu
        self.featurizer = get_featurizer(sample_rate=8000)
        self.audio_q = list()

    def save(self, waveforms, fname="wakeword_temp"):
        wf = wave.open(fname, "wb")
        # set the channels
        wf.setnchannels(1)
        # set the sample format
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        # set the sample rate
        wf.setframerate(8000)
        # write the frames as bytes
        wf.writeframes(b"".join(waveforms))
        # close the file
        wf.close()
        return fname

    def predict(self, audio):
        with torch.no_grad():
            fname = self.save(audio)
            waveform, _ = torchaudio.load(
                fname, normalization=False)  # don't normalize on train
            mfcc = self.featurizer(waveform).transpose(1, 2).transpose(0, 1)

            # TODO: read from buffer instead of saving and loading file
            # waveform = torch.Tensor([np.frombuffer(a, dtype=np.int16) for a in audio]).flatten()
            # mfcc = self.featurizer(waveform).transpose(0, 1).unsqueeze(1)

            out = self.model(mfcc)
            pred = torch.round(torch.sigmoid(out))
            return pred.item()

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) > 15:  # remove part of stream
                diff = len(self.audio_q) - 15
                for _ in range(diff):
                    self.audio_q.pop(0)
                action(self.predict(self.audio_q))
            elif len(self.audio_q) == 15:
                action(self.predict(self.audio_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                  args=(action,), daemon=True)
        thread.start()


class DemoAction:
    """action to take when wakeword is detected

        args: sensitivty. the lower the number the more sensitive the
        wakeword is to activation.
    """

    def __init__(self, sensitivity=10):
        # import stuff here to prevent engine.py from
        # importing unecessary modules during production usage
        import os
        import subprocess
        import random
        from os.path import join, realpath

        self.random = random
        self.subprocess = subprocess
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
    action = DemoAction(sensitivity=10)

    # action = lambda x: print(x)
    wakeword_engine.run(action)
    threading.Event().wait()
