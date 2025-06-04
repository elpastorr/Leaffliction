#!/bin/bash
if [ ! -d Apple ]; then
    curl -O https://cdn.intra.42.fr/document/document/17547/leaves.zip
    unzip leaves.zip
    rm leaves.zip
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
