#!/bin/bash
PROJECT_DIR="/data/project/cerebellum_ale"
filename=$1
n_subsamples=$2
subsample_size=$3

eval "$(conda shell.bash hook)" && \
conda activate ${PROJECT_DIR}/env_amin && \
${PROJECT_DIR}/env_amin/bin/python \
    ${PROJECT_DIR}/scripts/run_ale.py $filename $n_subsamples $subsample_size