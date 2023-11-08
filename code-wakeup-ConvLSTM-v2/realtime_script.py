
import sounddevice as sd
import numpy as np
import torch
import torch.optim as optim


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load scripted model and transform
scripted_model_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints-best/model.ptl"
scripted_transform_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM/checkpoints-best/transform.ptl"

model = torch.jit.load(scripted_model_path).to(device)
transform = torch.jit.load(scripted_transform_path)

model.eval()

def stream_audio(chunk_duration, samplerate=16000):
    """Stream audio in chunks."""
    with sd.InputStream(samplerate=samplerate, channels=1) as stream:
        while True:
            audio_chunk, _ = stream.read(int(samplerate * chunk_duration))
            yield audio_chunk

def process_chunk(audio_chunk):
    """Process an audio chunk and return model output probabilities."""
    # check count in audio_chunk

    audio_chunk_tensor = torch.tensor(audio_chunk).float()
    input_values = transform(audio_chunk_tensor)
    
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
    print(f"\r[{blocks}{spaces}] {probability:.10f}",) #end='' , flush=True)
    

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
            
        print_level(result[0].item())
        
        
