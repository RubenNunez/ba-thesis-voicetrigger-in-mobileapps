import sounddevice as sd
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# Record audio from the microphone
def record_audio(duration, samplerate=16000):
    print("Recording...")
    myrecording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=2, dtype='float64')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return myrecording

# Save the recording as a wav file
def save_wav(filename, recording, samplerate=16000):
    sf.write(filename, recording, samplerate)

# Main transcription function
def transcribe(filename):
    # Load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-base-960h")

    # Load the recorded audio file
    input_audio, _ = sf.read(filename, dtype="float32")
    
    # Make sure audio is mono (since we recorded with 2 channels)
    if len(input_audio.shape) > 1:
        input_audio = np.mean(input_audio, axis=1)
    
    # Tokenize
    input_values = processor(input_audio, return_tensors="pt", padding="longest").input_values
    
    # Retrieve logits
    logits = model(input_values).logits
    
    # Take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription

if __name__ == "__main__":
    recording = record_audio(10)  # Record for 10 seconds
    save_wav("recorded_audio.wav", recording)
    result = transcribe("recorded_audio.wav")
    print("Transcription:", result[0])

