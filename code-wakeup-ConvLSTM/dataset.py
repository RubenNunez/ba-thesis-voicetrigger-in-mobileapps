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
    def __init__(self, sample_rate=16000, n_fft=400, n_mels=128):
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            # This ensures that a square spectrogram is produced 128x128
            hop_length=int((sample_rate - n_fft) // (n_mels - 1)),
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80.0)

    def __call__(self, audio_waveform):
        # Check if audio waveform is shorter than 1 second (16000 samples)
        if audio_waveform.shape[1] < self.mel_spectrogram_transform.sample_rate:
            num_missing_samples = self.mel_spectrogram_transform.sample_rate - audio_waveform.shape[1]
            audio_waveform = torch.nn.functional.pad(audio_waveform, (0, num_missing_samples))

        audio_waveform = audio_waveform[:, :self.mel_spectrogram_transform.sample_rate]

        mel_spectrogram = self.mel_spectrogram_transform(audio_waveform)
        db_spectrogram = self.amplitude_to_db(mel_spectrogram)

        # Normalize the spectrogram
        eps = 1e-10
        db_spectrogram = (db_spectrogram - db_spectrogram.min()) / (db_spectrogram.max() - db_spectrogram.min() + eps)

        # Crop the spectrogram to 128x128 it was 128x132
        db_spectrogram = db_spectrogram[:, :, :128]

        # assert shape [1, 128, 128]
        assert db_spectrogram.shape == (1, 128, 128)
        
        return db_spectrogram
