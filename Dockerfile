FROM ubuntu:impish-20211015

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
    python3.7 \
    python3.7-dev \
    python3.7-venv \
    virtualenv \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER dnnf
WORKDIR /home/dnnf/

ENV MAKEFLAGS="--silent"

# load env on interactive shell
RUN echo "source .venv/bin/activate" >>.bashrc

COPY --chown=dnnf . .

RUN ./install.sh --include-cleverhans --include-foolbox --include-tensorfuzz --python python3.7 \
    && wget --progress=dot:giga http://cs.virginia.edu/~dls2fc/dnnf_benchmarks.tar.gz \
    && tar xf dnnf_benchmarks.tar.gz \
    && rm -f dnnf_benchmarks.tar.gz \
    && rm -rf .cache
