FROM ubuntu:focal

LABEL maintainer="David Shriver"

RUN useradd -ms /bin/bash dnnf

SHELL ["/bin/bash", "-c"]

RUN apt-get -qq update
RUN apt-get -qq install -y software-properties-common
RUN apt-get -qq install -y build-essential
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -qq update
RUN apt-get -qq install -y python3.7 python3.7-dev python3.7-venv

USER dnnf
WORKDIR /home/dnnf/

ENV MAKEFLAGS="--silent"

# load env on interactive shell
RUN echo "source .venv/bin/activate" >>.bashrc

COPY --chown=dnnf pyproject.toml .
COPY --chown=dnnf scripts/ scripts/
COPY --chown=dnnf install.sh .
COPY --chown=dnnf tools/ tools/
COPY --chown=dnnf dnnf/ dnnf/
COPY --chown=dnnf README.rst .

RUN ./install.sh
RUN wget http://cs.virginia.edu/~dls2fc/dnnf_benchmarks.tar.gz
RUN tar xf dnnf_benchmarks.tar.gz
