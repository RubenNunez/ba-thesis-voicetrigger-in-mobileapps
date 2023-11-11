import os
import glob
from pydub import AudioSegment

# progressbar import
from tqdm import tqdm

# Replace 'path_to_folders' with your specific folder path
base_path = '/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM-v2/other'

# Iterate over all .wav and .mp3 files
for audio_file in tqdm(glob.glob(os.path.join(base_path, '**/*'), recursive=True)):
    if audio_file.lower().endswith(('.wav', '.mp3')):
        audio = AudioSegment.from_file(audio_file)
        duration_in_seconds = len(audio) / 1000  # Duration in milliseconds
        if duration_in_seconds == 0:
            os.remove(audio_file)
            # print(f"Deleted {audio_file}")
