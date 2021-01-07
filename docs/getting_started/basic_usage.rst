Basic Usage
===========

DNNF can be run on correctness problems specified using ONNX for the network format and DNNP for the property format. DNNP is the same property specification language used by the DNNV_ verifier framework. A description of this specification language can be found in the `DNNV documentation`_.

To execute DNNF, first activate the virtual environment with::

  $ . .env.d/openenv.sh

This is only required if DNNF was installed from source. The virtual environment should open automatically if using the docker image or the provided VM.

The DNNF tool can then be run as follows::

  $ python -m dnnf PROPERTY --network NAME PATH

Where ``PROPERTY`` is the path to the property specification, ``NAME`` is the name of the network used in the property specification (typically ``N``), and ``PATH`` is the path to a DNN model in the ONNX_ format.

To see additional options, run::

  $ python -m dnnf -h
  usage: dnnf [-h] [-V] [--seed SEED] [-v | -q] [-N NETWORKS NETWORKS]
              [-p N_PROC] [-S N_STARTS] [--cuda]
              [--backend BACKEND [BACKEND ...]] [--set METHOD PARAM VALUE]
              property
  
  dnnf - deep neural network falsification
  
  positional arguments:
    property
  
  optional arguments:
    -h, --help            show this help message and exit
    -V, --version         show program's version number and exit
    --seed SEED           the random seed to use
    -v, --verbose         show messages with finer-grained information
    -q, --quiet           suppress non-essential messages
    -N, --network NETWORKS NETWORKS
    -p, --processors, --n_proc N_PROC
                          The maximum number of processors to use
    -S, --starts, --n_starts N_STARTS
                          The maximum number of random starts per sub-property
    --cuda                use cuda
    --backend BACKEND [BACKEND ...]
                          the falsification backends to use
    --set METHOD PARAM VALUE
                          set parameters for the falsification backend


Running on Benchmarks
^^^^^^^^^^^^^^^^^^^^^

We provide the property and network benchmarks used in the evaluation of our paper `here <http://cs.virginia.edu/~dls2fc/dnnf_benchmarks.tar.gz>`_.

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

If a property uses parameters, then the parameter value can be set using ``--prop.PARAMETER=VALUE``, e.g., ``--prop.epsilon=1``, similar to DNNV_.


.. _DNNV: https://github.com/dlshriver/DNNV
.. _`DNNV documentation`: https://dnnv.readthedocs.io/en/tacas21/usage/specifying_properties.html
.. _ONNX: https://onnx.ai
.. _cleverhans: https://github.com/tensorflow/cleverhans
