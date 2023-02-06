# Install TensorRT pip package
pip install --upgrade tensorrt==8.5.2.2

# Clone the repo
cd /workspace
git clone https://github.com/NVIDIA/TensorRT.git --branch 8.5.2 --single-branch
cd TensorRT
git submodule update --init --recursive

# Build libinfer plugin
mkdir -p build
cd build 
cmake .. -DTRT_OUT_DIR=$PWD/out
cd plugin
make -j$(nproc)