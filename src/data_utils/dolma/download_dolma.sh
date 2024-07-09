#!/bin/bash

DATA_DIR="src/data_utils/dolma/dolma_v1_7_sample"
PARALLEL_DOWNLOADS="16"
DOLMA_VERSION="sampled_urls"

# git clone https://huggingface.co/datasets/allenai/dolma
mkdir -p "${DATA_DIR}"


cat "src/data_utils/dolma/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR"