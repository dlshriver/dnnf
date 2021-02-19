#!/bin/bash
# set -x

ensure_exists() {
    if ! command -v $1 &>/dev/null; then
        echo "$1 could not be found. Please install before continuing."
        exit 1
    fi
}

ensure_exists "virtualenv"
ensure_exists "python2.7"
ensure_exists "python3.7"
ensure_exists "gcc"
ensure_exists "g++"
ensure_exists "wget"
ensure_exists "git"

rm -rf .venv/ bin/tensorfuzz/.venv bin/[^t]* bin/test* lib/ include/ info/ share/ libexec/

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
cd $PROJECT_DIR
mkdir -p bin
mkdir -p include
mkdir -p lib

cd bin
git clone https://github.com/dlshriver/tensorfuzz.git
cd tensorfuzz
git checkout a81df1b
virtualenv -p python2.7 .venv
. .venv/bin/activate
pip install "tensorflow>=1.6,<1.7" "numpy>=1.16,<1.17" "absl-py>=0.11,<0.12" "scipy>=1.2,<1.3" "pyflann>=1.6,<1.7" "onnx>=1.6,<1.7"
deactivate

cd $PROJECT_DIR
ln -s $PROJECT_DIR/bin/tensorfuzz/examples/localrobustness/tensorfuzz.sh bin/

virtualenv -p python3.7 .venv
. .venv/bin/activate
python -m pip install --upgrade pip setuptools flit

cd bin
git clone https://github.com/dlshriver/DNNV.git
cd DNNV
git checkout 893ea6e
flit install -s
cd $PROJECT_DIR
./scripts/install_neurify.sh
./scripts/install_eran.sh
./scripts/install_planet.sh
./scripts/install_reluplex.sh

cd $PROJECT_DIR
pip install "numpy>=1.18,<1.20" "onnx>=1.7,<1.8" "torch>=1.6,<1.7" "torchvision>=0.7,<0.8" "tensorflow>=1.15,<2.0" "pandas>=1.1<1.2"
pip install "cleverhans==3.0.1"
