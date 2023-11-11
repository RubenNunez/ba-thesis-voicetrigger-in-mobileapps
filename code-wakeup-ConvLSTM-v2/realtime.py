
import sounddevice as sd
import numpy as np
import torch
import torch.optim as optim

from model import WakeupTriggerConvLSTM2s
from dataset import AudioToSpectrogramTransform

# Load a checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss



device = "cuda" if torch.cuda.is_available() else "cpu"

model = WakeupTriggerConvLSTM2s(device=device).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

transform = AudioToSpectrogramTransform()

# Initialize model
checkpoint_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/checkpoints-v2/checkpoint_epoch_42_loss_0.010789588726497367.pt" 
model, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

model.eval()

def stream_audio(chunk_duration, samplerate=16000):
    """Stream audio in chunks."""
    with sd.InputStream(samplerate=samplerate, channels=1) as stream:
        while True:
            audio_chunk, _ = stream.read(int(samplerate * chunk_duration))
            yield audio_chunk

def process_chunk(audio_chunk):
    """Process an audio chunk and return model output probabilities."""

    # Ensure audio is mono
    if len(audio_chunk.shape) > 1:
        audio_chunk = np.mean(audio_chunk, axis=1)

    # Ensure audio is 1D array
    audio_chunk = audio_chunk.reshape(1, -1)
    audio_chunk_tensor = torch.tensor(audio_chunk).float()

    # Print audio chunk tensor info
    # print(audio_chunk_tensor.dtype)
    print(audio_chunk_tensor.min(), audio_chunk_tensor.max())

    # Transform audio
    input_values = transform(audio_chunk_tensor)
    input_values = input_values.unsqueeze(1)  # [batch_size, channels, height, width]
    input_values = input_values.to(device)

    # Retrieve logits and apply sigmoid activation to get probabilities
    with torch.no_grad():
        logits = model(input_values)
        probabilities = torch.sigmoid(logits).numpy()

    return probabilities

def print_level(probability):
    # Determine the number of blocks to display based on probability
    num_blocks = int(probability * 10)  # Using 10 blocks for full scale
    blocks = '█' * num_blocks
    spaces = ' ' * (10 - num_blocks)
    
    # Print the progress bar, overwrite the same line using \r
    print(f"\r[{blocks}{spaces}] {probability:.2f}", end='', flush=True)
    

if __name__ == "__main__":

    SAMPLE_RATE = 16000
    CHUNK_DURATION = 2  # seconds
    OVERLAP_DURATION = 0.1  # seconds
    overlap_buffer = np.zeros(int(SAMPLE_RATE * CHUNK_DURATION))  # Initialize with zeros 2s width

    for audio_chunk in stream_audio(OVERLAP_DURATION, samplerate=SAMPLE_RATE):
        audio_chunk = np.squeeze(audio_chunk)  # Convert to 1D array
    
        # Roll the overlap buffer to remove old audio and append new audio
        overlap_buffer = np.roll(overlap_buffer, shift=-len(audio_chunk))
        overlap_buffer[-len(audio_chunk):] = audio_chunk

        result = process_chunk(overlap_buffer)
                
        # Check if probability crosses a threshold
        print_level(result[0].item())
        # print(result[0].item())
        
