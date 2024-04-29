FROM ubuntu:22.04

# disable tzdata questions
ENV DEBIAN_FRONTEND=noninteractive

# use bash
SHELL ["/bin/bash", "-c"]

# install apt-utils
RUN apt-get update -y && \
  apt-get install -y apt-utils 2> >( grep -v 'debconf: delaying package configuration, since apt-utils is not installed' >&2 ) \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# essential tools
RUN apt-get update -y && apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  g++ \
  gdb \
  gfortran \
  liblapacke-dev \
  libmumps-dev \
  libopenblas-dev \
  libsuitesparse-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# copy files
COPY . /tmp/russell
WORKDIR /tmp/russell

# run tests
RUN cargo test
