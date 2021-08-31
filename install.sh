#!/bin/bash
set -e

python="python3"
extras=""
include_tensorfuzz=0

print_usage() {
    echo "Usage: install.sh [-h] [-p py |--python py] [--include-cleverhans] [--include-foolbox] [--include-tensorfuzz] [ENV_DIR]"
    echo ""
    echo "optional arguments:"
    echo "  ENV_DIR               directory to create the environment in (default: .venv)"
    echo "  -h, --help            show this help message and exit"
    echo "  -p, --python py       python version to use"
    echo "  --include-cleverhans  install cleverhans falsifiers"
    echo "  --include-foolbox     install foolbox falsifiers"
    echo "  --include-tensorfuzz  install tensorfuzz falsifier"
}

while [ -n "$1" ]; do # while loop starts
    case "$1" in
    -h | --help)
        print_usage
        exit 0
        ;;
    -p | --python)
        python=$2
        shift
        ;;
    --include-cleverhans)
        extras="$extras,cleverhans"
        ;;
    --include-foolbox)
        extras="$extras,foolbox"
        ;;
    --include-tensorfuzz)
        include_tensorfuzz=1
        ;;
    *)
        if [[ $# == 1 ]]; then
            break
        fi
        echo "Option $1 not recognized"
        print_usage
        exit 1
        ;;
    esac
    shift
done

env_dir=$(realpath "${1:-.venv}")

ensure_exists() {
    if ! command -v $1 &>/dev/null; then
        echo "$1 could not be found. Please install before continuing."
        if [ ! -z "$2" ]; then
            echo "$2"
        fi
        exit 1
    fi
}

if [ $include_tensorfuzz == 1 ]; then
    ensure_exists python2.7 "TensorFuzz requires python2.7"
    ensure_exists virtualenv "Installing TensorFuzz requires virtualenv"
fi
ensure_exists $python

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
echo $PROJECT_DIR

if [ "$VIRTUAL_ENV" != $env_dir ]; then
    if [ -e $env_dir/bin/activate ]; then
        echo "Using local virtual environment: $env_dir"
        . $env_dir/bin/activate
    else
        echo "Environment does not exist. Initializing..."
        $python -m venv $env_dir
        . $env_dir/bin/activate
    fi
else
    echo "Using active virtual environment: $VIRTUAL_ENV"
fi
cd $PROJECT_DIR
pip install --upgrade pip flit
if [ ! -z "$extras" ]; then
    extras="--extras=${extras:1}"
fi
flit install -s --deps production $extras

if [ $include_tensorfuzz == 1 ]; then
    mkdir -p $PROJECT_DIR/bin
    cd $PROJECT_DIR/bin
    rm -rf tensorfuzz
    rm -f $PROJECT_DIR/bin/tensorfuzz.sh

    git clone https://github.com/dlshriver/tensorfuzz.git
    cd tensorfuzz
    git checkout a81df1b
    
    virtualenv -p python2.7 .venv
    . .venv/bin/activate
    pip install "tensorflow>=1.6,<1.7" "numpy>=1.16,<1.17" "absl-py>=0.11,<0.12" "scipy>=1.2,<1.3" "pyflann>=1.6,<1.7" "onnx>=1.6,<1.7"
    deactivate

    cd $PROJECT_DIR
    ln -s $PROJECT_DIR/bin/tensorfuzz/examples/localrobustness/tensorfuzz.sh $PROJECT_DIR/bin/
fi
