#!/bin/bash
if [ ! -d images ]; then
    curl -O https://cdn.intra.42.fr/document/document/42144/leaves.zip
    unzip leaves.zip
    rm leaves.zip

fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
