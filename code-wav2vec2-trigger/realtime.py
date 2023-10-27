
import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torch.optim as optim

from model import TriggerWordWav2Vec2Model, config

# Load a checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss


device = "cuda" if torch.cuda.is_available() else "cpu"

model = TriggerWordWav2Vec2Model(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Initialize model and tokenizer once
checkpoint_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wav2vec2-trigger/checkpoints/checkpoint_epoch_8_loss_0.18035508620162163.pt" 
model, _, _, _ = load_checkpoint(checkpoint_path, model, optimizer)

processor = Wav2Vec2Processor.from_pretrained("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/wav2vec2-base-960h")

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

    # Tokenize
    input_values = processor(
        audio_chunk, 
        return_tensors="pt", 
        padding="longest", 
        sampling_rate=16000
    ).input_values

    if torch.cuda.is_available():
        input_values = input_values.to("cuda")

    # Retrieve logits and apply sigmoid activation to get probabilities
    with torch.no_grad():
        probabilities = model(input_values)
        probabilities = model(input_values).cpu().numpy()
    
    return probabilities


if __name__ == "__main__":
    CHUNK_DURATION = 1  # seconds
    OVERLAP_DURATION = 0.5  # seconds
    overlap_buffer = np.array([])

    for audio_chunk in stream_audio(CHUNK_DURATION):
        audio_chunk = np.squeeze(audio_chunk)  # Convert to 1D array
        audio_chunk_with_overlap = np.concatenate([overlap_buffer, audio_chunk])
        result = process_chunk(audio_chunk_with_overlap)
            
        # Check if probability crosses a threshold
        if result[0] > 0.8:  # Change this threshold as per your requirement
            print("Trigger detected!" + str(result))

        # Store overlap for next iteration
        overlap_buffer = audio_chunk[-int(OVERLAP_DURATION * 16000):]
