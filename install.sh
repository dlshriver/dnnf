#!/bin/sh
set -e

PROJECT_DIR=$(realpath $(dirname "$0"))
default_venv_dir=${VIRTUAL_ENV:-./.venv}

python="python3"
extras=""
include_tensorfuzz=0

print_usage() {
    echo "Usage: install.sh [-h] [-p py |--python py] [--include-cleverhans] [--include-foolbox] [--include-tensorfuzz]"
    echo ""
    echo "optional arguments:"
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
        echo "Option $1 not recognized"
        print_usage
        exit 1
        ;;
    esac
    shift
done

ensure_exists() {
    if ! type "$1" >/dev/null; then
        echo "$1 could not be found. Please install before continuing."
        if [ ! -z "$2" ]; then
            echo "$2"
        fi
        exit 1
    fi
}

if [ $include_tensorfuzz = 1 ]; then
    ensure_exists python2.7 "TensorFuzz requires python2.7"
    ensure_exists virtualenv "Installing TensorFuzz requires virtualenv"
    ensure_exists git "git is a required dependency"
fi
ensure_exists $python

read -p "Python virtual environment (default: $default_venv_dir):" venv_dir
if [ -z "$venv_dir" ]; then
    venv_dir=$default_venv_dir
fi
venv_dir=$(realpath $venv_dir)

if [ -e $venv_dir/bin/activate ]; then
    echo "Using local virtual environment: $venv_dir"
    . $venv_dir/bin/activate
else
    echo "Environment does not exist. Initializing..."
    $python -m venv $venv_dir
    . $venv_dir/bin/activate
fi

cd $PROJECT_DIR
pip install --upgrade pip
if [ ! -z "$extras" ]; then
    extras="[$(expr substr $extras 2 $(expr length $extras))]"
fi
pip install .$extras

if [ $include_tensorfuzz = 1 ]; then
    cd $venv_dir
    rm -rf tensorfuzz
    rm -f $venv_dir/bin/tensorfuzz.sh

    git clone https://github.com/dlshriver/tensorfuzz.git
    cd tensorfuzz
    git checkout a81df1b

    virtualenv -p python2.7 .venv
    . .venv/bin/activate
    pip install "tensorflow>=1.6,<1.7" "numpy>=1.16,<1.17" "absl-py>=0.11,<0.12" "scipy>=1.2,<1.3" "pyflann>=1.6,<1.7" "onnx>=1.6,<1.7"
    deactivate

    cd $PROJECT_DIR
    ln -s $venv_dir/tensorfuzz/examples/localrobustness/tensorfuzz.sh $venv_dir/bin/
fi
