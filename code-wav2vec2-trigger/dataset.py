import torch
import torchaudio

from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset, DataLoader

class TriggerWordDataset(Dataset):
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files
        self.labels = labels
        self.processor = Wav2Vec2Processor.from_pretrained("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/wav2vec2-base-960h")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_tensor = _load_audio(self.audio_files[idx])
        audio_input = self.processor(audio_tensor, return_tensors="pt", padding="longest", sampling_rate=16000).input_values
        label = self.labels[idx]
        return audio_input, label

def _load_audio(file_path, target_sample_rate=16000, max_length=24000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    
     # Truncate or pad the waveform to a fixed length
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    elif waveform.shape[1] < max_length:
        num_missing_samples = max_length - waveform.shape[1]
        last_dim_padding = (0, num_missing_samples)
        waveform = torch.nn.functional.pad(waveform, last_dim_padding)
    return waveform.squeeze().numpy()



def get_train_loader(audio_files, labels, batch_size=32):
    train_dataset = TriggerWordDataset(audio_files, labels)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
