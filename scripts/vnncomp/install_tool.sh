#!/bin/bash

set -e

TOOL_NAME="dnnf"
VERSION_STRING="v1"

if [ "$1" != "$VERSION_STRING" ]; then
    echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
    exit 1
fi

echo "Installing $TOOL_NAME"
DIR=$(dirname $(dirname $(dirname $(realpath $0))))

export DEBIAN_FRONTEND="noninteractive"
export TZ="America/New_York"

sudo apt-get update
sudo apt-get install -y software-properties-common
sudo apt-get install -y build-essential
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python2.7 python3.7 python3.7-dev python3-virtualenv python3.7-venv
sudo apt-get install -y psmisc # for killall, used in prepare_instance.sh script

$DIR/install.sh
