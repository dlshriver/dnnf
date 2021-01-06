#!/bin/bash

if [ -n "$PROJECT_DIR" ]; then
    echo "Closing open env: $PROJECT_ENV ($PROJECT_DIR)"
    . $PROJECT_DIR/.env.d/closeenv.sh
fi

envdir=$(dirname "${BASH_SOURCE[0]}")
currentproject=$PROJECT_ENV

. $envdir/env.sh

if [ "$currentproject" == "$PROJECT_ENV" ]; then
    . $envdir/closeenv.sh
    . $envdir/env.sh
fi
unset currentproject
unset envdir

append_path $PROJECT_DIR/bin/ PATH
append_path $PROJECT_DIR/lib/ LD_LIBRARY_PATH
append_path $PROJECT_DIR PYTHONPATH

if [ -e ./.venv/bin/activate ]; then
    . $PROJECT_DIR/.venv/bin/activate
else
    echo "Environment does not exist. Initializing..."
    $PROJECT_DIR/.env.d/initenv.sh
    . $PROJECT_DIR/.venv/bin/activate
fi

# eran paths
append_path $PROJECT_DIR/lib/eran/tf_verify PYTHONPATH
append_path $PROJECT_DIR/lib/ELINA/python_interface PYTHONPATH
append_path $PROJECT_DIR/lib/eran/ELINA/python_interface PYTHONPATH
