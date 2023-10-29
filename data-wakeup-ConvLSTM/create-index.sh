#!/bin/bash

# Define directories
BASE_DIR="/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM"
OTHER_LABEL_DIR="${BASE_DIR}/other"
FOOBY_LABEL_DIR="${BASE_DIR}/FOOBY"
SAVE_JSON_PATH="${BASE_DIR}"


# Execute the Python script
python ${BASE_DIR}/create_index.py \
    --other_label_dir "${OTHER_LABEL_DIR}" \
    --fooby_label_dir "${FOOBY_LABEL_DIR}" \
    --save_json_path "${SAVE_JSON_PATH}"

