FROM ubuntu:focal

LABEL maintainer="David Shriver"

RUN useradd -ms /bin/bash dnnf

SHELL ["/bin/bash", "-c"]

RUN apt-get -qq update
RUN apt-get -qq install -y software-properties-common
RUN apt-get -qq install -y build-essential
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -qq update
RUN apt-get -qq install -y python2.7 python3.7 python3.7-dev python3.7-venv virtualenv cmake wget git liblapack-dev openssl libssl-dev valgrind libtool

USER dnnf
WORKDIR /home/dnnf/

ENV MAKEFLAGS="--silent"

# load env on interactive shell
RUN echo "source .env.d/openenv.sh" >>.bashrc

# copy infrequently changed files to container
COPY --chown=dnnf pyproject.toml .
COPY --chown=dnnf .env.d/ .env.d/
COPY --chown=dnnf scripts/ scripts/
COPY --chown=dnnf install.sh .

RUN ./install.sh

COPY --chown=dnnf artifacts/ artifacts/
COPY --chown=dnnf tools/ tools/
COPY --chown=dnnf dnnf/ dnnf/
COPY --chown=dnnf README.md .
