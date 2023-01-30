FROM nvcr.io/nvidia/tensorrt:22.11-py3

# Install TensorRT with plugins
RUN pip install --upgrade pip && pip install --upgrade tensorrt

RUN git clone https://github.com/NVIDIA/TensorRT.git -b release/8.5 --single-branch \
    && cd TensorRT/ \
    && git submodule update --init --recursive

ENV TRT_OSSPATH=/workspace/TensorRT
WORKDIR /workspace/TensorRT

RUN mkdir -p build \
    && cd build \
    && cmake .. -DTRT_OUT_DIR=$PWD/out \
    && cd plugin \
    && make -j$(nproc)

# Set environment variables
ENV PLUGIN_LIBS="${TRT_OSSPATH}/build/out/libnvinfer_plugin.so"
ENV CUDA_MODULE_LOADING=LAZY

# Prepare the environment
WORKDIR /workspace/voltaML-fast-stable-diffusion

COPY requirements requirements

# Install python dependencies
RUN pip3 install -r requirements/api.txt
RUN pip3 install -r requirements/pytorch.txt
RUN pip3 install -r requirements/tensorrt.txt

COPY . /workspace/voltaML-fast-stable-diffusion

ENV LOG_LEVEL=INFO

# Run the server
RUN chmod +x start.sh

ENTRYPOINT ["bash", "./start.sh"]
