#!/bin/bash
PROJECT_DIR="/data/project/cerebellum_ale"
zmap_path=$1 

export PYTHONUTF8=1

source ${PROJECT_DIR}/venv/bin/activate && \
${PROJECT_DIR}/venv/bin/python \
    ${PROJECT_DIR}/scripts/run.py macm N N \
        -macm_in=$zmap_path