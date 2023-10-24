#!/bin/bash

# Define directories
BASE_DIR="/Users/ruben/Projects/ba-thesis-voicetrigger-in-mobileapps/data-wakeup-LSTM"
OTHER_LABEL_DIR="${BASE_DIR}/other"
HEY_FOOBY_LABEL_DIR="${BASE_DIR}/Hey_FOOBY"
SAVE_JSON_PATH="${BASE_DIR}"


# Execute the Python script
python ${BASE_DIR}/create_index.py \
    --other_label_dir "${OTHER_LABEL_DIR}" \
    --hey_fooby_label_dir "${HEY_FOOBY_LABEL_DIR}" \
    --save_json_path "${SAVE_JSON_PATH}"

