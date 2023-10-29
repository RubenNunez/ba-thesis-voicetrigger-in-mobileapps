#!/bin/bash

# Define directories
BASE_DIR="/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-ConvLSTM"
DIRS=("FOOBY" "Hello_FOOBY" "Hey_FOOBY" "Hi_FOOBY" "OK_FOOBY" "other" "other")

# Function to convert ogg to wav and then remove the ogg file
convert_ogg_to_wav() {
    local dir_path="$1"
    for file in "$dir_path"/*.ogg; do
        if [[ -f "$file" ]]; then
            local out_file="${file%.ogg}.wav"
            ffmpeg -i "$file" -y "$out_file"
            echo "Converted: $file -> $out_file"
            rm "$file"
            echo "Removed: $file"
        fi
    done
}

# Main script logic
for dir in "${DIRS[@]}"; do
    convert_ogg_to_wav "${BASE_DIR}/${dir}"
done

echo "Conversion and removal completed."
