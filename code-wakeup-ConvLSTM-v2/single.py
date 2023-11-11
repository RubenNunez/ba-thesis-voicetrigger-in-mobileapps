import torch
import torchaudio

from model import WakeupTriggerConvLSTM2s
from dataset import AudioToSpectrogramTransform

def print_level(probability):
    num_blocks = int(probability * 10)  # Using 10 blocks for full scale
    blocks = '█' * num_blocks
    spaces = ' ' * (10 - num_blocks)
    
    # Print the progress bar, overwrite the same line using \r
    print(f"\r[{blocks}{spaces}] {probability:.2f}", end='', flush=True)

def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


checkpoint_path = "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/checkpoints-v2/checkpoint_epoch_42_loss_0.010789588726497367.pt" 

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WakeupTriggerConvLSTM2s(device=device).to(device)
model = load_checkpoint(checkpoint_path, model)
model.eval()

transform = AudioToSpectrogramTransform()

#audio_waveform, sample_rate = torchaudio.load("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/FOOBY/FOOBY_1ffe7d0a46a04fe48a8bb3d8e0241ea9_998b96b8ae474fff8ef9e1b032f02a9f copy_volume_adjusted.wav")
audio_waveform, sample_rate = torchaudio.load("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/other/Baum_d88fb26508c94dadb48bab48808cd243_f5cc92ba43f84fd699d71478ffe84bd4_time_stretched.wav")
#audio_waveform, sample_rate = torchaudio.load("/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/silence/silence-3s.mp3")

# Ensure audio is mono
if len(audio_waveform.shape) > 1:
    audio_waveform = audio_waveform.mean(dim=1)


# Ensure audio is 1D array
audio_waveform = audio_waveform.reshape(1, -1)
audio_waveform_tensor = torch.tensor(audio_waveform).float()

# Transform audio
input_values = transform(audio_waveform_tensor)
input_values = input_values.unsqueeze(1)  # [batch_size, channels, height, width]
input_values = input_values.to(device)

# Retrieve logits and apply sigmoid activation to get probabilities
with torch.no_grad():
    logits = model(input_values)
    probabilities = torch.sigmoid(logits).numpy()

    print_level(probabilities[0].item())