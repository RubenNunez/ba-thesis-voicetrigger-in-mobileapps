import librosa
import librosa.display
import numpy as np
import os
import soundfile as sf

from tqdm import tqdm

class AudioAugment:
    def __init__(self, trigger_word_directory):
        self.trigger_word_directory = trigger_word_directory

    def _time_stretch(self, y, rate=1):
        return librosa.effects.time_stretch(y, rate)

    def _pitch_shift(self, y, sr, n_steps):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    def _add_noise(self, y, noise_level=0.005):
        noise = np.random.randn(len(y)) * noise_level
        return y + noise

    def _time_shift(self, y, sr, shift_max=2):
        shift = np.random.randint(-sr * shift_max, sr * shift_max)
        return np.roll(y, shift)

    def _volume_variation(self, y, low=0.7, high=1.3):
        return y * np.random.uniform(low, high)

    def augment_and_save(self):
        for filename in tqdm(os.listdir(self.trigger_word_directory)):
            file_path = os.path.join(self.trigger_word_directory, filename)
             # Skip non-audio files
            if not (filename.endswith('.wav') or filename.endswith('.mp3')):
                continue

            # Load the audio file with time stretching
            rate = np.random.uniform(0.85, 1.15)
            y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast', duration=2.5, offset=0.5)

            # Apply pitch shift
            y_pitch_shifted = self._pitch_shift(y, sr, n_steps=np.random.randint(-2, 2))

            # Apply noise
            y_with_noise = self._add_noise(y)

            # Apply time shift
            y_time_shifted = self._time_shift(y, sr)

            # Apply volume variation
            y_volume_adjusted = self._volume_variation(y)

            # Define the new filenames
            new_filenames = [
                filename.replace(".wav", "_time_stretched.wav"),
                filename.replace(".wav", "_pitch_shifted.wav"),
                filename.replace(".wav", "_with_noise.wav"),
                filename.replace(".wav", "_time_shifted.wav"),
                filename.replace(".wav", "_volume_adjusted.wav")
            ]

            augmented_audios = [
                y,
                y_pitch_shifted,
                y_with_noise,
                y_time_shifted,
                y_volume_adjusted
            ]

            # Save the augmented audio files beside the original ones
            for new_filename, augmented_audio in zip(new_filenames, augmented_audios):
                output_path = os.path.join(self.trigger_word_directory, new_filename)
                sf.write(output_path, augmented_audio, sr)


# Usage
augmentor = AudioAugment("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/other")
augmentor.augment_and_save()
