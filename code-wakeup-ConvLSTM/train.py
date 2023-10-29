import os
import numpy as np
from pathlib import Path

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

from tensorboardX import SummaryWriter

from model import WakeupTriggerConvLSTM
from dataset import get_train_loader

root_dir = Path("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WakeupTriggerConvLSTM(device=device).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-5) 
scheduler = StepLR(optimizer, step_size=10, gamma=0.01)

num_negative = 4080 # sum of all negative samples
num_positive = 1191  # sum of all positive samples
pos_weight = torch.tensor([num_negative / num_positive]).to(device)

# Assign this weight to the criterion
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

start_epoch = 0

# Load the checkpoint if one exists
# checkpoint_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints/checkpoint_epoch_66_loss_0.6524184110263983.pt"
# checkpoint = torch.load(checkpoint_path)
# 
# # Update model and optimizer with the saved states
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch']

# Data
audio_files = []
labels = []

for folder in root_dir.iterdir():
    if folder.is_dir():
        label = 1 if folder.name == "FOOBY" else 0
        for audio_file in folder.iterdir():
            if audio_file.suffix in ['.wav', '.mp3']:
                audio_files.append(str(audio_file))
                labels.append(label)

train_loader = get_train_loader(audio_files, labels)

writer = SummaryWriter('runs/training_logs')


epochs = 50
for epoch in range(epochs):
    total_loss = 0
    model.train()

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = batch
        inputs = inputs.squeeze(1)
        inputs, labels = inputs.to(device), labels.to(device)

        inputs = inputs.unsqueeze(1)  # [batch_size, channels, height, width]

        optimizer.zero_grad()
        # outputs = torch.sigmoid(model(inputs)).squeeze() # BCELoss
        outputs = model(inputs).squeeze() # BCEWithLogitsLoss

        loss = criterion(outputs, labels.float())
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # Log the batch loss
        writer.add_scalar('Batch Loss', loss.item(), (start_epoch + epoch + 1)*len(train_loader) + i)

    # Log the epoch loss
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Epoch Loss', avg_loss, (start_epoch + epoch + 1))

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    scheduler.step() # Step the scheduler

    # Save model checkpoint
    checkpoint_path = str(root_dir) + f"/checkpoints/checkpoint_epoch_{start_epoch + epoch + 1}_loss_{avg_loss}.pt"
    torch.save({
        'epoch': start_epoch + epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

writer.close()
