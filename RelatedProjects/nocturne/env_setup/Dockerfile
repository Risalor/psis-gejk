FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    CMAKE_POLICY_VERSION_MINIMUM=3.5

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libx11-dev \
    libxrandr-dev \
    libxcursor-dev \
    libxi-dev \
    libgl1-mesa-dev \
    libudev-dev \
    libopenal-dev \
    libvorbis-dev \
    libfreetype6-dev \
    libglew-dev \
    libflac-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --branch 2.5.x --depth 1 https://github.com/SFML/SFML.git && \
    cd SFML && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DSFML_BUILD_AUDIO=ON \
        -DSFML_BUILD_NETWORK=ON \
        -DSFML_BUILD_GRAPHICS=ON \
        -DSFML_BUILD_WINDOW=ON \
        -DSFML_USE_SYSTEM_DEPS=ON \
        -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/SFML

WORKDIR /app
RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install \
    hydra-core \
    omegaconf \
    gymnasium \
    pygame \
    numpy \
    matplotlib \
    tensorboard \
    wandb \
    tqdm \
    pytest \
    pre-commit \
    black \
    isort \
    flake8

COPY . /app/nocturne

WORKDIR /app/nocturne
RUN pip3 install -e . --no-build-isolation

WORKDIR /workspace

CMD ["/bin/bash"]