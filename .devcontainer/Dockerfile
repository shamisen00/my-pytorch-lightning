ARG ubuntu_version="20.04"

FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu${ubuntu_version}

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

WORKDIR /workspace

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
    uuid-dev \
    llvm-7-dev \
    && rm -rf /var/lib/apt/lists

RUN ln -s /usr/bin/llvm-config-7 /usr/bin/llvm-config

RUN wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz \
    && tar -xJf Python-3.9.0.tar.xz \
    && cd Python-3.9.0 \
    && ./configure --enable-optimizations \
    && make -j 8 \
    && make install

ENV PYTHONPATH=/workspace:$PYTHONPATH

ENV PATH /etc/poetry/bin:$PATH

RUN source ~/.bashrc \
    && curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 - 

ENV POETRY_VIRTUALENVS_CREATE=false

COPY ./pyproject.toml ./pyproject.toml
COPY ./poetry.lock ./poetry.lock

# Update and Install dependencies
RUN poetry lock --no-update && poetry install
