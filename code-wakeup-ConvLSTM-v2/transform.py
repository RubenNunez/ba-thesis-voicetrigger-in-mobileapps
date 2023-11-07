import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

class AudioToSpectrogramTransformJit(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=800, n_mels=128):
        super(AudioToSpectrogramTransformJit, self).__init__()
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=int((2 * sample_rate - n_fft) // (n_mels * 2 - 1)),
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80.0)

    def forward(self, audio_chunk):
        # Ensure audio is mono
        if len(audio_chunk.shape) > 1:
            audio_chunk = torch.mean(audio_chunk, dim=1)

        # Ensure audio is 1D array
        audio_chunk_tensor = audio_chunk.float().reshape(1, -1)

        # Here Starts the Transform
        # -----------------------------

        # Check if audio waveform is shorter than 2 second (16000 * 2 samples)
        if audio_chunk_tensor.shape[1] < self.mel_spectrogram_transform.sample_rate * 2:
            num_missing_samples = (self.mel_spectrogram_transform.sample_rate * 2) - audio_chunk_tensor.shape[1]
            audio_chunk_tensor = torch.nn.functional.pad(audio_chunk_tensor, (0, num_missing_samples))

        audio_chunk_tensor = audio_chunk_tensor[:, :self.mel_spectrogram_transform.sample_rate * 2]

        mel_spectrogram = self.mel_spectrogram_transform(audio_chunk_tensor)
        db_spectrogram = self.amplitude_to_db(mel_spectrogram)

        # Normalize the spectrogram
        eps = 1e-10
        db_spectrogram = (db_spectrogram - db_spectrogram.min()) / (db_spectrogram.max() - db_spectrogram.min() + eps)

        # Crop the spectrogram to 128x128 it was 128x132
        db_spectrogram = db_spectrogram[:, :, :256]

        # assert shape [1, 128, 256]
        assert db_spectrogram.shape == (1, 128, 256)
        
        return db_spectrogram.unsqueeze(1)  # [batch_size, channels, height, width]
