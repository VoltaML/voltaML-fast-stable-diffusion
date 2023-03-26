# Local Installation

This guide will walk you through the process of installing the application locally on your system without using Docker.

## Windows

### Install Ubuntu WSL

1. Install Ubuntu from the Microsoft Store.

2. Launch Ubuntu and follow the prompts to finish the setup.

### Run Automated Setup Script

```bash
curl -fsSLO https://raw.githubusercontent.com/VoltaML/voltaML-fast-stable-diffusion/experimental/scripts/wsl-install.sh
chmod +x wsl-install.sh
. wsl-install.sh
```

## Linux

Please refer to the [PyTorch installation guide](/developers/pytorch), better installation instructions for Linux are coming soon.
