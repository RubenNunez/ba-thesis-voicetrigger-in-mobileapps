#!/bin/bash

mkdir wav2vec2-base-960h

# Install git lfs and clone the model
git lfs install
git clone https://huggingface.co/facebook/wav2vec2-base-960h ./wav2vec2-base-960h

# Delete the .git FOLDER
rm -rf ./wav2vec2-base-960h/.git

echo "Model successfully downloaded."
