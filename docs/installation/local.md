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

### How to start the application after closing it

::: tip
I have an easier way planned for the future, but for now, you'll have to run the following commands every time you want to start the application.
:::

1. Launch Ubuntu from the Start Menu
2. `cd voltaML-fast-stable-diffusion` (or wherever you cloned the repository)
3. `source venv/bin/activate`
4. `python main.py`

## Linux

Please refer to the [PyTorch installation guide](/developers/pytorch), better installation instructions for Linux are coming soon.

AITemplate can be installed by following the steps [here](https://github.com/facebookincubator/AITemplate#installation)
Please note that this is tesed on Ubuntu systems and that it may not work on other Linux distributions.
