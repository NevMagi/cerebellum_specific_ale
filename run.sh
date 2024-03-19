#!/bin/bash
PROJECT_DIR="/data/project/cerebellum_ale"
analysis=$1 # "ALE" or "SALE"
bd=$2
subbd=${3:-"All"}
n_subsamples=${4:-"0"}
subsample_size=${5:-"50"}

# subsample_size=${subsample_size//_/.}
echo "subsample_size = $subsample_size"

export PYTHONUTF8=1

source ${PROJECT_DIR}/venv/bin/activate && \
${PROJECT_DIR}/venv/bin/python \
    ${PROJECT_DIR}/scripts/run.py $analysis $bd $subbd \
        -n_subsamples=$n_subsamples -subsample_size=$subsample_size