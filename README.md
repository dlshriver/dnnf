# Reducing DNN Properties to Enable Falsification with Adversarial Attacks

This repo accompanies the paper [Reducing DNN Properties to Enable Falsification with Adversarial Attacks](https://davidshriver.me/files/publications/ICSE21-DNNF.pdf), and provides a tool for running falsification methods such as adversarial attacks on DNN property specifications specified using the [DNNP](https://dnnv.readthedocs.io/en/latest/usage/specifying_properties.html) language of [DNNV](https://github.com/dlshriver/DNNV). For an overview of our paper, check out our [video presentation](https://youtu.be/hcQFYUFwp_U).

Additional documentation can be found on [Read the Docs](https://dnnf.readthedocs.io/).

## Install

We provide instructions for installing DNNF with pip, installing DNNF from source, as well as for building and running a docker image.

### Pip Install

DNNF can be installed using pip by running:

```bash
  $ pip install dnnf
```

This will install the latest release of DNNF on [PyPI](https://pypi.org/project/dnnf/).
To install the optional falsification backends, you can replace `dnnf` in the above command with `dnnf[BACKENDS]`, 
where `BACKENDS` is a comma separated list of the backends you wish to include (i.e., `cleverhans` or `foolbox`).
To install the most recent changes from GitHub, run:

```bash
  $ pip install git+https://github.com/dlshriver/dnnf.git@main
```

To install the cleverhans or foolbox backends, run the above command with the option `--install-option="--extras-require=cleverhans,foolbox"` included.

> Installation with pip will not install the TensorFuzz falsification backend. Currently this backend is only available through manual installation or the provided docker image.

### Source Install

The required dependencies to install DNNF from source are:

- python3
- git

The additional, optional tensorfuzz backend also requires:

- python2.7
- virtualenv

If you do not plan to use tensorfuzz, then these dependencies are not required.
Please ensure that the required dependencies are installed prior to running the installation script.
For example, on a fresh Ubuntu 20.04 system, the dependencies can be installed using apt as follows:

```bash
  $ sudo add-apt-repository ppa:deadsnakes/ppa
  $ sudo apt-get update
  $ sudo apt-get install git python3.8 # python2.7 virtualenv
```

To install DNNF in the local directory, download this repo and run the provided installation script,
optionally specifying which backends to include during installation:

```bash
  $ ./install.sh [--include-cleverhans] [--include-foolbox] [--include-tensorfuzz]
```

To see additional installation options, use the `-h` option.

We have successfully tested this installation procedure on machines running Ubuntu 20.04 and CentOS 7.

### Docker Install

We provide a pre-built docker image containing DNNF, available on [Docker Hub](https://hub.docker.com/r/dlshriver/dnnf). To use this image, run the following:

```bash
  $ docker pull dlshriver/dnnf
  $ docker run --rm -it dlshriver/dnnf
  (.venv) dnnf@hostname:~$ dnnf -h
```

To build a docker image with the latest changes to DNNF, run:

```bash
  $ docker build . -t dlshriver/dnnf
  $ docker run --rm -it dlshriver/dnnf
  (.venv) dnnf@hostname:~$ dnnf -h
```

## Execution

DNNF can be run on correctness problems specified using [ONNX](https://onnx.ai) and [DNNP](https://docs.dnnv.org/en/stable/dnnp/introduction.html). 
DNNP is the same property specification language used by the [DNNV](https://github.com/dlshriver/dnnv) verifier framework. 
A description of this specification language can be found in the [DNNV documentation](https://docs.dnnv.org/en/stable/dnnp/introduction.html).

To execute DNNF, first activate the virtual environment with:

```bash
  $ . .venv/bin/activate
```

This is only required if DNNF was installed manually. The virtual environment should open automatically if using the docker image.

The DNNF tool can then be run as follows:

```bash
  $ dnnf PROPERTY --network NAME PATH
```

Where `PROPERTY` is the path to the property specification, `NAME` is the name of the network used in the property specification (typically `N`), and `PATH` is the path to a DNN model in the [ONNX](https://onnx.ai) format.

To see additional options, run:

```bash
  $ dnnf -h
```

To see the currently available falsification backends, use the `--long-help` option.


### Running on the Benchmarks

We provide several DNN verification benchmarks in DNNP and ONNX formats in [dlshriver/dnnv-benchmarks](https://github.com/dlshriver/dnnv-benchmarks). 
This benchmark repository includes both the DNNF-GHPR and the DNNF-CIFAR-EQ benchmarks introduced by DNNF!

To execute DNNF on a problem in one of the benchmarks, 
first navigate to the desired benchmark directory in `benchmarks` (e.g., `DNNF-GHPR`, `DNNF-GHPR`). 
Then run DNNF as specified above. 
For example, to run DNNF with the Projected Gradient Descent adversarial attack from [cleverhans](https://github.com/tensorflow/cleverhans) on an DNNF-GHPR property and network,
run:

```bash
  $ cd benchmarks/DNNF-GHPR
  $ dnnf properties/dronet_property_0.py --network N onnx/dronet.onnx --backend cleverhans.projected_gradient_descent
```

Which will produce output similar to:

```bash
  Falsifying: Forall(x, (((0 <= x) & (x <= 1) & (N[(slice(2, -3, None), 1)](x) <= -2.1972245773362196)) ==> ((-0.08726646259971647 <= N[(slice(2, -1, None), 0)](x)) & (N[(slice(2, -1, None), 0)](x) <= 0.08726646259971647))))

  dnnf
    result: sat
    falsification time: 0.6901
    total time: 2.3260
```

The available backends for falsification are:

  - [CleverHans](https://github.com/tensorflow/cleverhans)
    
    - `cleverhans.carlini_wagner_l2`
    - `cleverhans.fast_gradient_method`
    - `cleverhans.hop_skip_jump_attack`
    - `cleverhans.projected_gradient_descent`
    - `cleverhans.spsa`

  - [FoolBox](https://github.com/bethgelab/foolbox)

    - `foolbox.ATTACK` where `ATTACK` is the name of an adversarial attack from 
      [this list](https://foolbox.readthedocs.io/en/stable/modules/attacks.html#module-foolbox.attacks)

  - [TensorFuzz](https://github.com/brain-research/tensorfuzz)

    - ``tensorfuzz``

Attack specific parameters can be set using the `--set BACKEND NAME VALUE` option.
For example, to set the `nb_iter` parameter of the `cleverhans.projected_gradient_descent` attack to 40 steps,
you can specify `--set cleverhans.projected_gradient_descent nb_iter 40`.

If a property uses parameters, then the parameter value can be set using `--prop.PARAMETER=VALUE`, 
e.g., `--prop.epsilon=1`, similar to DNNV.


## Acknowledgements

This material is based in part upon work supported by the National Science Foundation under grant number 1900676 and 2019239.
