#!/bin/bash

MODEL_PATH="/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM/checkpoints/optimized/optimized_model.pt"
SENSITIVITY=10

python engine.py --model_file $MODEL_PATH --sensitivity $SENSITIVITY