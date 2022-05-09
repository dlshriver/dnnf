FROM ubuntu:focal

LABEL maintainer="David Shriver"

RUN useradd -ms /bin/bash dnnf

SHELL ["/bin/bash", "-c"]

ENV TZ=America/New_York

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends -y \
    git \
    python2.7 \
    python2.7-dev \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    virtualenv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER dnnf
WORKDIR /home/dnnf/

# turn off tensorflow logging
ENV TF_CPP_MIN_LOG_LEVEL="3"

# load env on interactive shell
RUN echo "source .venv/bin/activate" >>.bashrc

# copy dnnf to the image
COPY --chown=dnnf . .

# install dnnf and benchmarks
RUN echo | ./install.sh --include-cleverhans --include-foolbox --include-tensorfuzz --python python3.8 \
    && git clone https://github.com/dlshriver/dnnv-benchmarks.git \
    && ln -s /home/dnnf/dnnv-benchmarks/benchmarks artifacts \
    && rm -rf .cache
