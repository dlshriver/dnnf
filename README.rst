Reducing DNN Properties to Enable Falsification with Adversarial Attacks
========================================================================

This repo accompanies the paper `Reducing DNN Properties to Enable Falsification with Adversarial Attacks <https://davidshriver.me/publications/>`_, and provides a tool for running falsification methods such as adversarial attacks on DNN property specifications specified using the DNNP_ language of DNNV_.

Additional documentation can be found on `Read the Docs`_.

Install
-------

We provide instructions for manually installing DNNF, as well as for building and running a docker image.

Manual Install
^^^^^^^^^^^^^^

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
Both machines used gcc version 7.


Docker Install
^^^^^^^^^^^^^^

DNNF can be installed and run using docker as follows::

  $ docker build . -t dlshriver/dnnf
  $ docker run -it dlshriver/dnnf
  (.venv) dnnf@hostname:~$ python -m dnnf -h


Execution
---------

To execute DNNF, first activate the virtual environment with::

  $ . .env.d/openenv.sh

This is only required if DNNF was installed manually. The virtual environment should open automatically if using the docker image.

The DNNF tool can then be run as follows::

  $ python -m dnnf PROPERTY --network NAME PATH

Where ``PROPERTY`` is the path to the property specification, ``NAME`` is the name of the network used in the property specification (typically ``N``), and ``PATH`` is the path to a DNN model in the ONNX_ format.

To see additional options, run::

  $ python -m dnnf -h


Running on the Benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^^

We provide the property and network benchmarks used in our evaluation `here <http://cs.virginia.edu/~dls2fc/dnnf_benchmarks.tar.gz>`_.

To execute DNNF on a problem in one of the benchmarks, first navigate to the desired benchmark directory in ``artifacts`` (i.e., ``acas_benchmark``, ``neurifydave_benchmark``, or ``ghpr_benchmark``). Then run DNNF as specified above. For example, to run DNNF with the Projected Gradient Descent adversarial attack from `cleverhans`_ on an ACAS property and network, run::

  $ cd artifacts/acas_benchmark
  $ python -m dnnf properties/property_2.py --network N onnx/N_3_1.onnx --backend cleverhans.ProjectedGradientDescent

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

Because we use DNNV_ to run verifiers, in order to run a verifier on a problem in one of the benchmarks, please read the instructions in the DNNV_ repository.
As an example, to run the ERAN deepzono verifier on the same ACAS property and network as above, run::

  $ cd artifacts/acas_benchmark
  $ python -m dnnv onnx/N_3_1.onnx properties/property_2.py --network N onnx/N_3_1.onnx --eran

Which should produce output similar to::

  Verifying Network:
  Input_0                         : Input([1 5], dtype=float32)
  Gemm_0                          : Gemm(Input_0, ndarray(shape=(50, 5)), ndarray(shape=(50,)))
  Relu_0                          : Relu(Gemm_0)
  Gemm_1                          : Gemm(Relu_0, ndarray(shape=(50, 50)), ndarray(shape=(50,)))
  Relu_1                          : Relu(Gemm_1)
  Gemm_2                          : Gemm(Relu_1, ndarray(shape=(50, 50)), ndarray(shape=(50,)))
  Relu_2                          : Relu(Gemm_2)
  Gemm_3                          : Gemm(Relu_2, ndarray(shape=(50, 50)), ndarray(shape=(50,)))
  Relu_3                          : Relu(Gemm_3)
  Gemm_4                          : Gemm(Relu_3, ndarray(shape=(50, 50)), ndarray(shape=(50,)))
  Relu_4                          : Relu(Gemm_4)
  Gemm_5                          : Gemm(Relu_4, ndarray(shape=(50, 50)), ndarray(shape=(50,)))
  Relu_5                          : Relu(Gemm_5)
  Gemm_6                          : Gemm(Relu_5, ndarray(shape=(5, 50)), ndarray(shape=(5,)))

  Verifying property:
  Forall(x0, ((([[ 0.6 -0.5 -0.5  0.45 -0.5 ]] <= x0) & (x0 <= [[ 0.68 0.5  0.5  0.5 -0.45]])) ==> (numpy.argmax(N(x0)) != 0)))
  ...
  dnnv.verifiers.eran
    result: unknown
    time: 2.5711


Running the Evaluation
^^^^^^^^^^^^^^^^^^^^^^

To run the full evaluation in our paper (WARNING: this may take several hundred hours), run::

  $ scripts/run_all.sh

This script will sequentially run all falsifiers and verifiers on all benchmarks.
It will save results in the ``results/`` directory, as comma separated values files.
There will be one file for each method and benchmark variant.
These files can be combined into a single csv by running the following in the root directory::

  $ python tools/combine_results.py

Which will generate a file called ``results.csv`` in the current directory.
This CSV file will have 6 columns:

- ``Artifact`` specifies the artifact being run, e.g., ACAS Xu
- ``Variant`` specifies a variant of the artifact, e.g., DroNet or MNIST for GHPR
- ``ProblemId`` specifies an identifier for the problem being checked
- ``Method`` specifies the method used to check the problem
- ``Result`` specifies the result of falsification or verification
- ``TotalTime`` specifies the time to generate a result

If you have access to a cluster with slurm, execution may be sped up by running script ``scripts/run_all_slurm.sh``, which will launch slurm jobs rather than running each technique sequentially.

Troubleshooting
---------------

If any of the tools fail to run, these steps may help to identify and fix the issue:

- First ensure that the directory ``.venv/`` was created in the root of this directory.
- If this directory does not exist then virtualenv was likely not installed or could not be found by the installation script. 
  Try re-installing virtualenv and ensure it is visible on the execution path, then run ``./install.sh`` again.
- If virtualenv is installed but the ``.venv/`` directory is still missing, then python3.7 may not have been found by the installation script. 
  Try re-installing python3.7 and ensure it is visible on the execution path, then run ``./install.sh`` again.
- If one of the verifiers fails to run because the executable could not be found, 
  try installing the verifier again with the verifier specific installation script in the ``scripts/`` directory (e.g., ``./scripts/install_neurify.sh`` to install neurify).



.. _DNNV: https://github.com/dlshriver/DNNV
.. _DNNP: https://dnnv.readthedocs.io/en/tacas21/usage/specifying_properties.html
.. _ONNX: https://onnx.ai
.. _cleverhans: https://github.com/tensorflow/cleverhans
.. _`Read the Docs`: https://dnnf.readthedocs.io/
