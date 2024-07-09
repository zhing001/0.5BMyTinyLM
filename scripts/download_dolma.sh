#!/bin/bash

DATA_DOWNLOAD_DIR="cache/dolma_sample"
PARALLEL_DOWNLOADS="1"
DOLMA_VERSION="sampled_urls"

# git clone https://huggingface.co/datasets/allenai/dolma
mkdir -p "${DATA_DOWNLOAD_DIR}"


cat "cache/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DOWNLOAD_DIR"