#!/bin/bash
PROJECT_DIR="/data/project/cerebellum_ale"
analysis=$1 # "ALE" or "SALE"
bd=$2
subbd=${3:"Main"}
n_subsamples=${4:0}
subsample_size=${5:0}
n_iters=${6:10000}

export PYTHONUTF8=1

source ${PROJECT_DIR}/venv/bin/activate && \
${PROJECT_DIR}/venv/bin/python \
    ${PROJECT_DIR}/scripts/run.py $analysis $bd \
        -subbd=$subbd -n_subsamples=$n_subsamples \
        -subsample_size=$subsample_size -n_iters=$n_iters