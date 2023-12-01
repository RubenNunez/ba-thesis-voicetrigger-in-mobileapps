
import sounddevice as sd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

import psutil
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

scripted_model_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/checkpoints-best/model.ptl"
scripted_transform_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/checkpoints-best/transform.ptl"
model = torch.jit.load(scripted_model_path).to(device)
transform = torch.jit.load(scripted_transform_path)
model.eval()

# For taking average of the cpu usage lets define a list to store the cpu usage values
# lets do this also for the time taken to infer the model
cpu_usage_list = []
time_taken_list = []

def stream_audio(chunk_duration, samplerate=16000):
    """Stream audio in chunks."""
    with sd.InputStream(samplerate=samplerate, channels=1) as stream:
        while True:
            audio_chunk, _ = stream.read(int(samplerate * chunk_duration))
            yield audio_chunk

def process_chunk(audio_chunk):
    """Process an audio chunk and return model output probabilities."""
    start_time = time.time()

    audio_chunk_tensor = torch.tensor(audio_chunk).float()
    input_values = transform(audio_chunk_tensor)
    
    with torch.no_grad():
        logits = model(input_values)
        probabilities = torch.sigmoid(logits).numpy()

    end_time = time.time()
    cpu_usage = psutil.cpu_percent()

    cpu_usage_list.append(cpu_usage)
    time_taken_list.append(end_time - start_time)

    return probabilities

def print_level(probability):
    num_blocks = int(probability * 10)
    blocks = '█' * num_blocks
    spaces = ' ' * (10 - num_blocks)
    print(f"\r[{blocks}{spaces}] {probability:.10f}",) #end='' , flush=True)
    
if __name__ == "__main__":
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 2  # seconds
    OVERLAP_DURATION = 0.1  # seconds
    overlap_buffer = np.zeros(int(SAMPLE_RATE * CHUNK_DURATION))  # Initialize with zeros 2s width
    
    start_time = time.time()

    try:
        for audio_chunk in stream_audio(OVERLAP_DURATION, samplerate=SAMPLE_RATE):
            audio_chunk = np.squeeze(audio_chunk)  # Convert to 1D array
        
            # Roll the overlap buffer to remove old audio and append new audio
            overlap_buffer = np.roll(overlap_buffer, shift=-len(audio_chunk))
            overlap_buffer[-len(audio_chunk):] = audio_chunk

            result = process_chunk(overlap_buffer)
            print_level(result[0].item())

            # Calculate the elapsed time
            elapsed_time = time.time() - start_time
            if elapsed_time >= 10:
                break

    except KeyboardInterrupt:
        print('\nDone')

pd.DataFrame(cpu_usage_list).to_csv('profiling_cpu_usage.csv')
pd.DataFrame(time_taken_list).to_csv('profiling_time_taken.csv')

time_axis = [i * 0.1 for i in range(len(cpu_usage_list))]

# Create a plot of profiling results
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(time_axis, cpu_usage_list, label='CPU Usage')
plt.title('CPU Usage Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('CPU Usage (%)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_axis, time_taken_list, label='Time Taken')
plt.title('Processing Time Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Time Taken (seconds)')
plt.legend()

plt.tight_layout()
plt.savefig('profiling_plot.png')
plt.show()
    


