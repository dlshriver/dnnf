#!/bin/bash
# set -x

ensure_exists() {
    if ! command -v $1 &>/dev/null; then
        echo "$1 could not be found. Please install before continuing."
        exit 1
    fi
}

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
echo $PROJECT_DIR

if [ "$VIRTUAL_ENV" == "" ]; then
    if [ -e $PROJECT_DIR/.venv/bin/activate ]; then
        echo "Using local virtual environment: $PROJECT_DIR/.venv"
        . $PROJECT_DIR/.venv/bin/activate
    else
        echo "Environment does not exist. Initializing..."
        python3.7 -m venv $PROJECT_DIR/.venv
        . $PROJECT_DIR/.venv/bin/activate
    fi
else
    echo "Using active virtual environment: $VIRTUAL_ENV"
fi
cd $PROJECT_DIR
pip install --upgrade pip flit setuptools
flit install -s
