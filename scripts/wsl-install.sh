# /bin/bash

echo Please paste your HuggingFace token here:
read hftoken

echo ----Updating and upgrading system---
sudo apt update && sudo apt upgrade -y

echo ----Cloning voltaML---
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion.git --branch experimental
cd voltaML-fast-stable-diffusion

echo ----Adding CUDA repo---
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
sudo add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /"
sudo apt-get update

echo ----Installing CUDA, Python, ROCmInfo---
sudo apt install -y python3.10 python3.10-venv build-essential cuda python3-pip rocminfo

echo ----Adding required paths and env vars to .bashrc---
echo "" >> ~/.bashrc
echo export CUDA_HOME="/usr/local/cuda-12.1" >> ~/.bashrc
echo export LD_LIBRARY_PATH=/lib/wsl/lib:\$LD_LIBRARY_PATH >> ~/.bashrc
echo export PATH="\$PATH:\$CUDA_HOME/bin" >> ~/.bashrc
echo export HUGGINGFACE_TOKEN=$hftoken >> ~/.bashrc

echo ----Reloading .bashrc---
source ~/.bashrc

echo ----Verifying CUDA installation---
nvcc -V

echo ----Creating a virtual environment---
python3 -m venv venv
source venv/bin/activate

echo ----Installing AITemplate---
git clone --recursive https://github.com/facebookincubator/AITemplate
cd AITemplate/python
pip install wheel
python3 setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
cd ../..

echo ----Installing voltaML---
python3 main.py