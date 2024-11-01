#!/bin/bash

cd $(dirname $0)

# create Python environemnt in ../venv
if [ ! -d "../venv" ]; then
    python3 -m venv ../venv
fi

# install dependencies
source ../venv/bin/activate
pip install -r requirements.txt

# fix nimare v0.2.0 encoding issue
python fix_nimare_encoding.py