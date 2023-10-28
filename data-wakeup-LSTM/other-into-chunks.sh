#!/bin/bash

# Define directories
BASE_DIR="/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM"
OTHER_LABEL_DIR="${BASE_DIR}/other"

SCRIPT_PATH="${BASE_DIR}/other_into_chunks.py"
SAVE_PATH="${BASE_DIR}/other_chunked"

# Ensure the save path exists
mkdir -p $SAVE_PATH

# Loop through all the audio files in the OTHER_LABEL_DIR and split them into chunks
for AUDIO_FILE in ${OTHER_LABEL_DIR}/*; do
    python $SCRIPT_PATH --seconds 10 --audio_file_name $AUDIO_FILE --save_path $SAVE_PATH
done

# Print completion message
echo "All audio files have been chunked!"
