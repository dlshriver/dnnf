Reducing DNN Properties to Enable Falsification with Adversarial Attacks
========================================================================

This repo accompanies the paper `Reducing DNN Properties to Enable Falsification with Adversarial Attacks <https://davidshriver.me/files/publications/ICSE21-DNNF.pdf>`_, and provides a tool for running falsification methods such as adversarial attacks on DNN property specifications specified using the DNNP_ language of DNNV_. For an overview of our paper, check out our `video presentation <https://youtu.be/hcQFYUFwp_U>`_.

Additional documentation can be found on `Read the Docs`_.

Install
-------

We provide instructions for installing DNNF with pip, installing DNNF from source, as well as for building and running a docker image.

Pip Install
^^^^^^^^^^^

DNNF can be installed using pip by running::

  $ pip install dnnf

This will install the last version uploaded to `PyPI`_. To install the most recent changes from GitHub, run::

  $ pip install git+https://github.com/dlshriver/DNNF.git@main

*Note:* installation with pip will not install the TensorFuzz falsification backend. Currently this backend is only available through manual installation or the provided docker image.

Source Install
^^^^^^^^^^^^^^

The required dependencies for installation from source are:

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

Docker Install
^^^^^^^^^^^^^^

We provide a pre-built docker image containing DNNF, available on `Docker Hub`_. To use this image, run the following::

  $ docker pull dlshriver/dnnv
  $ docker run -it dlshriver/dnnf
  (.venv) dnnf@hostname:~$ dnnf -h

To build a docker image with the latest changes to DNNF, run::

  $ docker build . -t dlshriver/dnnf
  $ docker run -it dlshriver/dnnf
  (.venv) dnnf@hostname:~$ dnnf -h


Execution
---------

To execute DNNF, first activate the virtual environment with::

  $ . .venv/bin/activate

This is only required if DNNF was installed manually. The virtual environment should open automatically if using the docker image.

The DNNF tool can then be run as follows::

  $ dnnf PROPERTY --network NAME PATH

Where ``PROPERTY`` is the path to the property specification, ``NAME`` is the name of the network used in the property specification (typically ``N``), and ``PATH`` is the path to a DNN model in the ONNX_ format.

To see additional options, run::

  $ dnnf -h


Running on the Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^

We provide the property and network benchmarks used in our evaluation `here <http://cs.virginia.edu/~dls2fc/dnnf_benchmarks.tar.gz>`_.

To execute DNNF on a problem in one of the benchmarks, first navigate to the desired benchmark directory in ``artifacts`` (i.e., ``acas_benchmark``, ``neurifydave_benchmark``, or ``ghpr_benchmark``). Then run DNNF as specified above. For example, to run DNNF with the Projected Gradient Descent adversarial attack from `cleverhans`_ on an ACAS property and network, run::

  $ cd artifacts/acas_benchmark
  $ dnnf properties/property_2.py --network N onnx/N_3_1.onnx --backend cleverhans.ProjectedGradientDescent

Which will produce output similar to::

  Falsifying: Forall(x0, (((x0 <= [[ 0.68 0.5  0.5  0.5 -0.45]]) & ([[ 0.6 -0.5 -0.5  0.45 -0.5 ]] <= x0)) ==> (numpy.argmax(N(x0)) != 0)))

  dnnf
    result: sat
    time: 2.6067

The available backends for falsification are:

- ``cleverhans.LBFGS``, which also requires setting parameters ``--set cleverhans.LBFGS y_target "[[-1.0, 0.0]]"``
- ``cleverhans.BasicIterativeMethod``
- ``cleverhans.FastGradientMethod``
- ``cleverhans.DeepFool``, which also requires setting parameters ``--set cleverhans.DeepFool nb_candidate 2``
- ``cleverhans.ProjectedGradientDescent``
- ``tensorfuzz``

If a property uses parameters, then the parameter value can be set using ``--prop.PARAMETER=VALUE``, e.g., ``--prop.epsilon=1``, similar to DNNV_.


Acknowledgements
----------------

This material is based in part upon work supported by the National Science Foundation under grant number 1900676 and 2019239.


.. _DNNV: https://github.com/dlshriver/DNNV
.. _DNNP: https://dnnv.readthedocs.io/en/tacas21/usage/specifying_properties.html
.. _ONNX: https://onnx.ai
.. _cleverhans: https://github.com/tensorflow/cleverhans
.. _`Read the Docs`: https://dnnf.readthedocs.io/
.. _`PyPI`: https://pypi.org/project/dnnf/
.. _`Docker Hub`: https://hub.docker.com/r/dlshriver/dnnf
