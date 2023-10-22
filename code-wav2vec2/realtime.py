import sounddevice as sd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# Initialize model and tokenizer once
processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-base-960h")

def stream_audio(chunk_duration, samplerate=16000):
    """Stream audio in chunks."""
    with sd.InputStream(samplerate=samplerate, channels=1) as stream:
        while True:
            audio_chunk, _ = stream.read(int(samplerate * chunk_duration))
            yield audio_chunk

def transcribe_chunk(audio_chunk):
    """Transcribe an audio chunk."""
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

    # Retrieve logits and decode
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription

if __name__ == "__main__":
    CHUNK_DURATION = 1  # seconds
    OVERLAP_DURATION = 0.5  # seconds
    overlap_buffer = np.array([])

    for audio_chunk in stream_audio(CHUNK_DURATION):
        audio_chunk = np.squeeze(audio_chunk)  # Convert to 1D array
        audio_chunk_with_overlap = np.concatenate([overlap_buffer, audio_chunk])
        result = transcribe_chunk(audio_chunk_with_overlap)
            
        # Print the latest word from the transcription
        words = result[0].split()
        if words:
            print(words[-1])

        # Store overlap for next iteration
        overlap_buffer = audio_chunk[-int(OVERLAP_DURATION * 16000):]

