FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as tensorrt

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install software-properties-common -y \
    && add-apt-repository --yes ppa:deadsnakes/ppa

RUN apt update && apt install build-essential curl git-core -y

RUN apt install tensorrt=8.5.3.1-1+cuda11.8 tensorrt-dev=8.5.3.1-1+cuda11.8 tensorrt-libs=8.5.3.1-1+cuda11.8 -y

RUN git clone https://github.com/NVIDIA/TensorRT /TensorRT \
    && cd /TensorRT \
    && git submodule update --init --recursive

WORKDIR /TensorRT

RUN curl https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.sh -L -o ./install_cmake \
    && chmod +x ./install_cmake \
    && mkdir -p /opt/cmake \
    && ./install_cmake --skip-license --prefix="/opt/cmake" \
    && ln -s /opt/cmake/bin/* /usr/bin \
    && ls -al /opt/cmake

RUN mkdir -p build && cd build \
    && cmake .. -DTRT_OUT_DIR=$PWD/out \
    && cd plugin \
    && make -j$(nproc)


FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 as main

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt install curl -y

RUN apt install tensorrt=8.5.3.1-1+cuda11.8 -y

RUN curl -sL https://deb.nodesource.com/setup_18.x | bash

RUN apt install nodejs -y

RUN npm i -g yarn

RUN cd  /usr/local/cuda-11.8/targets/x86_64-linux/lib/ \
    && ln -s libcublas.so.11 libcublas.so.12 \
    && ln -s libcublasLt.so.11 libcublasLt.so.12

ENV LD_LIBRARY_PATH /usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

WORKDIR /app

COPY --from=tensorrt /TensorRT/build/out/libnvinfer_plugin.so.8 /app/lib/trt/lib/libnvinfer_plugin.so

COPY .. /app

RUN pwd && ls -al && sleep 30

RUN cd frontend && yarn install && yarn build

RUN pip3 install requirements/tensorrt.txt

ENTRYPOINT [ "python", "main.py" ]
