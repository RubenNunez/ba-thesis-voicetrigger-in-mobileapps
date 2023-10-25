#!/bin/bash

# Define the base directory for the project
BASE_DIR="/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM"

# Change to the code directory to run the training script
cd "/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/code-wakeup-LSTM"

# Start training using the python script
python train.py \
    --sample_rate 8000 \
    --epochs 100 \
    --batch_size 32 \
    --eval_batch_size 1 \
    --lr 0.1 \
    --model_name "wakeup" \
    --train_data_json "$BASE_DIR/train.json" \
    --test_data_json "$BASE_DIR/test.json" \
    --save_checkpoint_path "$BASE_DIR/checkpoints/" \
    --hidden_size 32
