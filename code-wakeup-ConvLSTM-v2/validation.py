
import os

from dataset import AudioToSpectrogramTransform
from model import WakeupTriggerConvLSTM2s
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sounddevice as sd
import soundfile as sf

from dataset import get_data_loader

import torch
from torchaudio.transforms import Resample
from torch.utils.data import DataLoader, TensorDataset

from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/eval_logs-v2')

# metrics
def get_metrics_for_logits(logits, labels, inputs):
    # Convert the logits to probabilities
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).float()

    correct_predictions = torch.eq(predictions, labels.float()).sum().item()
    batch_accuracy = correct_predictions / inputs.size(0)

    TP = (predictions * labels.float()).sum().item()
    FP = (predictions * (1 - labels.float())).sum().item()
    FN = ((1 - predictions) * labels.float()).sum().item()

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)   

    batch_f1_score = 2 * (precision * recall) / (precision + recall + 1e-10) 
    
    return batch_accuracy, batch_f1_score


root_dir = Path("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/test-data/")
silence_file_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/silence/silence-3s.mp3"


audio_files = []
labels = []

for folder in root_dir.iterdir():
    if folder.is_dir():
        if folder.name != "FOOBY" and folder.name != "other":
            continue
        label = 1 if folder.name == "FOOBY" else 0
        for audio_file in folder.iterdir():
            if audio_file.suffix in ['.wav', '.mp3']:
                audio_files.append(str(audio_file))
                labels.append(label)

                if label == 0: # add silence file for negative samples
                    audio_files.append(silence_file_path)
                    labels.append(label)

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/checkpoints-best/checkpoint_epoch_133_loss_0.04460274705179358.pt")
model = WakeupTriggerConvLSTM2s(device).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
transform = AudioToSpectrogramTransform()

eval_loader = get_data_loader(audio_files, labels, batch_size=32)

def run(model, eval_loader, device):
    model.eval()
    total_accuracy = 0
    total_f1_score = 0

    with torch.no_grad():
       for i, (inputs, labels) in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs).squeeze()
            batch_accuracy, batch_f1_score = get_metrics_for_logits(logits, labels, inputs)

            writer.add_scalar('Test/Batch Accuracy', batch_accuracy, i)
            writer.add_scalar('Test/Batch F1 Score', batch_f1_score, i)
            total_accuracy += batch_accuracy
            total_f1_score += batch_f1_score

    avg_accuracy = total_accuracy / len(eval_loader)
    avg_f1_score = total_f1_score / len(eval_loader)

    writer.add_scalar('Test/Average Accuracy', avg_accuracy, 0)
    writer.add_scalar('Test/Average F1 Score', avg_f1_score, 0)

    return avg_accuracy, avg_f1_score

# Run the test loop
test_accuracy, test_f1_score = run(model, eval_loader, device)
print(f"Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1_score}")

writer.close()