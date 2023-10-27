import torch
import torch.optim as optim
import torch.nn as nn


from model import TriggerWordWav2Vec2Model, config
from dataset import get_train_loader


device = "cuda" if torch.cuda.is_available() else "cpu"

model = TriggerWordWav2Vec2Model(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss

audio_files = [...]  
labels = [...] 

train_loader = get_train_loader(audio_files, labels)

epochs = 3
for epoch in range(epochs):
    total_loss = 0
    model.train()
    
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")
