#!/bin/bash

VERSION_STRING="v1"

if [ "$1" != "$VERSION_STRING" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

CATEGORY="$2"
ONNX_FILE="$3"
VNNLIB_FILE="$4"
RESULTS_FILE="$5"
TIMEOUT="$6"

RESULTS_DIR="$(dirname $RESULTS_FILE)"
TMP_RESULTS_FILE="${RESULTS_DIR}/tmp_$(basename $RESULTS_FILE)"

echo "Running benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

DIR=$(dirname $(dirname $(dirname $(realpath $0))))
. $DIR/.venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=3

timeout $TIMEOUT dnnf "$VNNLIB_FILE" --network N "$ONNX_FILE" --vnnlib --n_start=1000 >$TMP_RESULTS_FILE

exitcode=$?
if [ $exitcode -eq 0 ]; then
    grep "result: sat" $TMP_RESULTS_FILE
    issat=$?
    grep "result: unknown" $TMP_RESULTS_FILE
    isunknown=$?
    if [ $issat -eq 0 ]; then
        echo "violated" >$RESULTS_FILE
    elif [ $isunknown -eq 0 ]; then
        echo "unknown" >$RESULTS_FILE
    fi
elif [ $exitcode -eq 124 ]; then
    echo "timeout" >$RESULTS_FILE
else
    echo "error" >$RESULTS_FILE
fi

rm $TMP_RESULTS_FILE
