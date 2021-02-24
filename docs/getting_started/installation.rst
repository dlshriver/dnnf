.. _installation:

Installation
============

DNNF can be installed from source or using docker.
We also provide a `pre-configured VirtualBox VM <http://TODO>`_, containing the tool and data used for the evaluation in `Reducing DNN Properties to Enable Falsification with Adversarial Attacks <https://davidshriver.me/publications/>`_.

From Source
-----------

The required dependencies are:

- git
- wget
- gcc-7
- g++-7
- virtualenv
- python3.7
- python3.7-dev
- python2.7

Please ensure that these dependencies are installed prior to running the rest of the installation script.
For example, on a fresh Ubuntu 20.04 system, the dependencies can be installed using apt as follows::

  $ sudo add-apt-repository ppa:deadsnakes/ppa
  $ sudo apt-get update
  $ sudo apt-get install python3.7
  $ sudo apt-get install python3.7-dev
  $ sudo apt-get install python2.7
  $ sudo apt-get install virtualenv
  $ sudo apt-get install gcc-7
  $ sudo apt-get install g++-7

To install DNNF, and the tools used to run the study to the local directory, run the provided installation script::

  $ ./install.sh

This may take several minutes and there may be several prompts during installation.
We have successfully tested this installation procedure on machines running Ubuntu 20.04 and CentOS 7.
Both machines used gcc version 7, but other versions will likely work as well.

Using Docker
------------

DNNF can be installed and run using docker as follows::

  $ docker build . -t dlshriver/dnnf
  $ docker run -it dlshriver/dnnf
  (.venv) dnnf@hostname:~$ python -m dnnf -h


Using Pip
---------

DNNF can also be installed using pip as follows:

  $ pip install dnnf
