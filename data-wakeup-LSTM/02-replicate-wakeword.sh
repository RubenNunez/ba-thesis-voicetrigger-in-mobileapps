#!/bin/bash

BASE_DIR="/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM"
WAKEWORDS_DIR="${BASE_DIR}/Hey_FOOBY"

# directory where the replicated clips should be saved, creating it if it doesn't exist
COPY_DEST="${BASE_DIR}/Hey_FOOBY_replicated"
mkdir -p $COPY_DEST

# number of replications for each clip
COPY_NUMBER=5

python replicate_wakeword.py --wakewords_dir $WAKEWORDS_DIR --copy_destination $COPY_DEST --copy_number $COPY_NUMBER
