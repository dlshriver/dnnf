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
ensure_exists "git"

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
echo $PROJECT_DIR

if [ "$VIRTUAL_ENV" == "" ]; then
    if [ -e $PROJECT_DIR/.venv/bin/activate ]; then
        echo "Using local virtual environment: $PROJECT_DIR/.venv"
        . $PROJECT_DIR/.venv/bin/activate
    else
        echo "Environment does not exist. Initializing..."
        virtualenv -p python3.7 $PROJECT_DIR/.venv
        . $PROJECT_DIR/.venv/bin/activate
    fi
else
    echo "Using active virtual environment: $VIRTUAL_ENV"
fi
cd $PROJECT_DIR
pip install --upgrade pip flit setuptools
flit install -s

echo "#!/bin/bash" >$VIRTUAL_ENV/bin/tensorfuzz.sh
echo "cd $VIRTUAL_ENV/opt/tensorfuzz/" >>$VIRTUAL_ENV/bin/tensorfuzz.sh
echo ". .venv/bin/activate" >>$VIRTUAL_ENV/bin/tensorfuzz.sh
echo "examples/localrobustness/tensorfuzz.sh $@" >>$VIRTUAL_ENV/bin/tensorfuzz.sh
chmod u+x $VIRTUAL_ENV/bin/tensorfuzz.sh

rm -rf $VIRTUAL_ENV/opt/tensorfuzz/
mkdir -p $VIRTUAL_ENV/opt/
cd $VIRTUAL_ENV/opt/
git clone https://github.com/dlshriver/tensorfuzz.git
cd tensorfuzz
git checkout a81df1b
virtualenv -p python2.7 .venv
. .venv/bin/activate
pip install "tensorflow>=1.6,<1.7" "numpy>=1.16,<1.17" "absl-py>=0.11,<0.12" "scipy>=1.2,<1.3" "pyflann>=1.6,<1.7" "onnx>=1.6,<1.7"
deactivate
