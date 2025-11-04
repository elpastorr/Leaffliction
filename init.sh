#!/bin/bash
if [ ! -d images ]; then
    curl -O https://cdn.intra.42.fr/document/document/17547/leaves.zip
    unzip leaves.zip
    rm leaves.zip

fi

if [ ! -d images/Apple ]; then
    mkdir images/Apple images/Grape
    mv images/Apple_* images/Apple/.
    mv images/Grape_* images/Grape/.
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt
