#!/bin/bash

CHECKPOINTS_DIR="/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM/checkpoints"
OPTIMIZED_MODEL_DIR="$CHECKPOINTS_DIR/optimized"

# Ensure the output directory exists
mkdir -p "$OPTIMIZED_MODEL_DIR"

# Use ls to list all checkpoint files and sort them by name (which includes the date and time)
# The tail -n 1 command selects the last (most recent) file in the sorted list
NEWEST_CHECKPOINT=$(ls -t "$CHECKPOINTS_DIR" | grep '^wakeup_' | head -1)

# Construct the full path to the newest checkpoint file
NEWEST_CHECKPOINT_PATH="$CHECKPOINTS_DIR/$NEWEST_CHECKPOINT"

# Construct the path for the optimized model
SAVE_PATH="$OPTIMIZED_MODEL_DIR/optimized_model.pt"

# Run the optimization script with the newest checkpoint
python optimize.py --model_checkpoint "$NEWEST_CHECKPOINT_PATH" --save_path "$SAVE_PATH"
