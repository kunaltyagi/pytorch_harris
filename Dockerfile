FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# build-depends of OpenVINO... bit much, could be trimmed down
# didn't include visualization libs
RUN apt update && \
    apt install -y \
    automake \
    autoconf \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    g++-multilib \
    gcc-multilib \
    git \
    git-lfs \
    gstreamer1.0-plugins-base \
    libavcodec-dev \
    libavformat-dev \
    libboost-regex-dev \
    libglib2.0-dev \
    libgstreamer1.0-0 \
    libopenblas-dev \
    libpng-dev \
    libssl-dev \
    libswscale-dev \
    libtool \
    libusb-1.0-0-dev \
    pkg-config \
    python3 \
    python3-pip \
    unzip \
    unzip \
    wget && \
    rm -fr /var/lib/apt/lists

ARG NUM_JOBS=6
RUN git clone https://github.com/openvinotoolkit/openvino/ --depth=1 --recursive && \
    mkdir -p openvino/build && \
    cd openvino/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CPPLINT:=OFF -DENABLE_FASTER_BUILD:=ON .. && \
    make -j ${NUM_JOBS} -l ${NUM_JOBS} install && \
    cd ../.. && \
    rm -fr openvino

RUN pip install -U pip setuptools wheel \
    defusedxml requests networkx \
    torch torchvision onnx \
    openvino
