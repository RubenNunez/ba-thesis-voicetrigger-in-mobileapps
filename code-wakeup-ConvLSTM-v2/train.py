import os
import numpy as np
from pathlib import Path

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

from tensorboardX import SummaryWriter

from model import WakeupTriggerConvLSTM2s
from dataset import get_train_loader

# metrics
def get_metrics_for_logits(logits):
    # Convert the logits to probabilities
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.5).float()

    # Calculate the accuracy for this batch
    correct_predictions = torch.eq(predictions, labels.float()).sum().item()
    batch_accuracy = correct_predictions / inputs.size(0)

    # Calculate true positives, false positives, and false negatives
    TP = (predictions * labels.float()).sum().item()
    FP = (predictions * (1 - labels.float())).sum().item()
    FN = ((1 - predictions) * labels.float()).sum().item()

    # Calculate precision and recall
    precision = TP / (TP + FP + 1e-10) # adding a small value to avoid division by zero
    recall = TP / (TP + FN + 1e-10)    # adding a small value to avoid division by zero

    # Calculate F1 score for this batch
    batch_f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)  # adding a small value to avoid division by zero
    
    return batch_accuracy, batch_f1_score


root_dir = Path("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WakeupTriggerConvLSTM2s(device=device).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0001) 
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
print(f"the model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

num_negative = 26000     # sum of all negative samples
num_positive = 10000    # sum of all positive samples
pos_weight = torch.tensor([num_negative / num_positive]).to(device)

# Assign this weight to the criterion
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

start_epoch = 0

# Load the checkpoint if one exists
# checkpoint_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints-v2/checkpoint_epoch_66_loss_0.6524184110263983.pt"
# checkpoint = torch.load(checkpoint_path)
# 
# # Update model and optimizer with the saved states
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch']

# Data
audio_files = []
labels = []

silence_file_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/silence/silence-3s.mp3"

for folder in root_dir.iterdir():
    if folder.is_dir():
        label = 1 if folder.name == "FOOBY" else 0
        for audio_file in folder.iterdir():
            if audio_file.suffix in ['.wav', '.mp3']:
                audio_files.append(str(audio_file))
                labels.append(label)
                
                if label == 0: # add silence file for negative samples
                    audio_files.append(silence_file_path)
                    labels.append(label)

train_loader = get_train_loader(audio_files, labels)

writer = SummaryWriter('runs/training_logs-v2')

epochs = 100
for epoch in range(epochs):
    model.train()

    total_loss = 0
    total_accuracy = 0
    total_f1_score = 0

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
        # [batch_size, channels, height, width]
        inputs, labels = batch        
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # outputs = torch.sigmoid(model(inputs)).squeeze() # BCELoss
        logits = model(inputs).squeeze() # BCEWithLogitsLoss
        loss = criterion(logits, labels.float())

        loss.backward()
        optimizer.step()

        # Log the batch loss
        batch_accuracy, batch_f1_score = get_metrics_for_logits(logits)
        total_loss += loss.item()
        total_accuracy += batch_accuracy
        total_f1_score += batch_f1_score

        # Log the metrics to TensorBoardX
        writer.add_scalar('Batch Loss', loss.item(), (start_epoch + epoch + 1)*len(train_loader) + i)
        writer.add_scalar('Batch Accuracy', batch_accuracy, (start_epoch + epoch + 1)*len(train_loader) + i)
        writer.add_scalar('Batch F1 Score', batch_f1_score, (start_epoch + epoch + 1)*len(train_loader) + i)
        

    # Log the epoch loss
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    avg_f1_score = total_f1_score / len(train_loader)

    writer.add_scalar('Epoch Loss', avg_loss, (start_epoch + epoch + 1))
    writer.add_scalar('Epoch Accuracy', avg_accuracy, (start_epoch + epoch + 1))
    writer.add_scalar('Epoch F1 Score', avg_f1_score, (start_epoch + epoch + 1))
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss} Accuracy: {avg_accuracy} F1 Score: {avg_f1_score}")

    scheduler.step() # Step the scheduler

    # Save model checkpoint
    checkpoint_path = str(root_dir) + f"/checkpoints-v2/checkpoint_epoch_{start_epoch + epoch + 1}_loss_{avg_loss}.pt"
    torch.save({
        'epoch': start_epoch + epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

writer.close()
