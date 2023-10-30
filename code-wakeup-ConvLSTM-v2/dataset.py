import torch
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import Dataset, DataLoader


class WakeupTriggerDataset(Dataset):
    def __init__(self, audio_files, labels, transform=None):
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_waveform, sample_rate = torchaudio.load(self.audio_files[idx])

        assert audio_waveform.shape[1] > 0, f"Empty waveform for file: {self.audio_files[idx]}"

        
        if sample_rate != self.transform.mel_spectrogram_transform.sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.transform.mel_spectrogram_transform.sample_rate)
            audio_waveform = resampler(audio_waveform)
        
        if self.transform:
            audio_waveform = self.transform(audio_waveform)

        if torch.isnan(audio_waveform).any():
            print(f"NaN values in transformed audio for file: {self.audio_files[idx]}")

        label = self.labels[idx]
        return audio_waveform, label

def get_train_loader(audio_files, labels, batch_size=32):
    transform = AudioToSpectrogramTransform()
    train_dataset = WakeupTriggerDataset(audio_files, labels, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class AudioToSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_fft=800, n_mels=128):
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            # This ensures that a square spectrogram is produced 128x256
            hop_length=int((2 * sample_rate - n_fft) // (n_mels * 2 - 1)),
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80.0)

    def __call__(self, audio_waveform):
        # Check if audio waveform is shorter than 2 second (16000 samples)
        if audio_waveform.shape[1] < self.mel_spectrogram_transform.sample_rate * 2:
            num_missing_samples = self.mel_spectrogram_transform.sample_rate - audio_waveform.shape[1]
            audio_waveform = torch.nn.functional.pad(audio_waveform, (0, num_missing_samples))

        audio_waveform = audio_waveform[:, :self.mel_spectrogram_transform.sample_rate * 2]

        mel_spectrogram = self.mel_spectrogram_transform(audio_waveform)
        db_spectrogram = self.amplitude_to_db(mel_spectrogram)

        # Normalize the spectrogram
        eps = 1e-10
        db_spectrogram = (db_spectrogram - db_spectrogram.min()) / (db_spectrogram.max() - db_spectrogram.min() + eps)

        # Crop the spectrogram to 128x128 it was 128x132
        db_spectrogram = db_spectrogram[:, :, :256]

        # assert shape [1, 128, 256]
        assert db_spectrogram.shape == (1, 128, 256)
        
        return db_spectrogram

# 
# import matplotlib.pyplot as plt
# import torchaudio
# 
#         
# transform = AudioToSpectrogramTransform()
# waveform, sample_rate = torchaudio.load("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/FOOBY/FOOBY_3bc31366e8af480483d580d16e9870db_3b3995e89cf742788cbe192b94f3aaae copy_time_shifted_pitch_shifted.wav")
# 
# if len(waveform.shape) == 1:
#     waveform = waveform.unsqueeze(0)
# 
# db_spectrogram = transform(waveform)
# db_spectrogram = db_spectrogram.squeeze(0)
# 
# # Plot and save the spectrogram
# plt.figure(figsize=(10, 4))
# plt.imshow(db_spectrogram.numpy(), cmap='viridis', origin='lower', aspect='auto')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel-scaled Spectrogram')
# plt.ylabel('Mel Bin')
# plt.xlabel('Time Frame')
# plt.tight_layout()
# plt.savefig('spectrogram.png')
# plt.show()
