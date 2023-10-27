
from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset, DataLoader

class TriggerWordDataset(Dataset):
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files
        self.labels = labels
        self.processor = Wav2Vec2Processor.from_pretrained("../wav2vec2-base-960h")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_input = self.processor(self.audio_files[idx], return_tensors="pt", padding="longest").input_values
        label = self.labels[idx]
        return audio_input, label


def get_train_loader(audio_files, labels, batch_size=32):
    train_dataset = TriggerWordDataset(audio_files, labels)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
