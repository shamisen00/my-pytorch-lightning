ARG ubuntu_version="20.04"

FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu${ubuntu_version}

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN apt update && apt install -y \
    wget \
    git \
    curl \
    tk-dev \
    build-essential \
    libbz2-dev libdb-dev \
    libreadline-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    libssl-dev \
    zlib1g-dev \
    uuid-dev

# Install the key
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
#     && dpkg -i cuda-keyring_1.0-1_all.deb

# RUN apt update && apt install -y \
#     cuda-drivers \
#     cuda \
#     && rm -rf /var/lib/apt/lists

RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tar.xz \
    && tar -xJf Python-3.10.0.tar.xz \
    && cd Python-3.10.0 \
    && ./configure --enable-optimizations \
    && make -j 8 \
    && make install

ENV PATH /etc/poetry/bin:$PATH

RUN source ~/.bashrc \
    && curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 - 

ENV POETRY_VIRTUALENVS_CREATE=false
