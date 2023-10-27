import os
from pathlib import Path

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

from tensorboardX import SummaryWriter

from model import TriggerWordWav2Vec2Model, config
from dataset import get_train_loader

# Load a checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss


root_dir = Path("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wav2vec2-trigger") 
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TriggerWordWav2Vec2Model(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss
 
# loaded_checkpoint_path = str(root_dir) + "/checkpoints/checkpoint_epoch_8_loss_0.3730935366993601.pt"
# model, optimizer, start_epoch, last_loss = load_checkpoint(loaded_checkpoint_path, model, optimizer)

start_epoch = 0

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

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    model.train()

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = batch
        inputs = inputs.squeeze()
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log the batch loss
        writer.add_scalar('Batch Loss', loss.item(), epoch*len(train_loader) + i)

    # Log the epoch loss
    avg_loss = total_loss/len(train_loader)
    writer.add_scalar('Epoch Loss', avg_loss, epoch)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

    # Save model checkpoint
    checkpoint_path = str(root_dir) + f"/checkpoints/checkpoint_epoch_{start_epoch + epoch+1}_loss_{avg_loss}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

writer.close()
