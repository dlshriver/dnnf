.. _installation:

Installation
============

DNNF can be installed using pip, manually from source, or using docker.
We also provide a `pre-configured VirtualBox VM <https://doi.org/10.5281/zenodo.4439219>`_, containing the tool and data used for the evaluation in `Reducing DNN Properties to Enable Falsification with Adversarial Attacks <https://doi.org/10.1109/ICSE43902.2021.00036>`_ (`pre-print`_).

Pip Install
-----------

DNNF can be installed using pip by running::

  $ pip install dnnf

This will install the last version uploaded to `PyPI`_. To install the most recent changes from GitHub, run::

  $ pip install git+https://github.com/dlshriver/DNNF.git@main

*Note:* installation with pip will not install the TensorFuzz falsification backend. Currently this backend is only available through manual installation or the provided docker image.

Source Install
--------------

The required dependencies to install DNNF from source are:

- git
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
  $ sudo apt-get install git

To install DNNF in the local directory, download this repo and run the provided installation script::

  $ ./install.sh

We have successfully tested this installation procedure on machines running Ubuntu 20.04 and CentOS 7.

Using Docker
------------

We provide a pre-built docker image containing DNNF, available on `Docker Hub`_. To use this image, run the following::

  $ docker pull dlshriver/dnnv
  $ docker run -it dlshriver/dnnf
  (.venv) dnnf@hostname:~$ dnnf -h

To build a docker image with the latest changes to DNNF, run::

  $ docker build . -t dlshriver/dnnf
  $ docker run -it dlshriver/dnnf
  (.venv) dnnf@hostname:~$ dnnf -h


.. _`pre-print`: <https://davidshriver.me/files/publications/ICSE21-DNNF.pdf
.. _`PyPI`: https://pypi.org/project/dnnf/
.. _`Docker Hub`: https://hub.docker.com/r/dlshriver/dnnf
