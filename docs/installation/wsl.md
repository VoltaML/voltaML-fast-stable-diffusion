# WSL Installation

## Backends

- ✅ Backend available and supported
- ❌ Backend not available yet
- 🚧 Backend is in the development or testing phase

| Backend    | Supported |
| ---------- | --------- |
| PyTorch    | ✅        |
| AITemplate | ✅        |
| ONNX       | 🚧        |

## Installation

### Install Ubuntu WSL

1. Install Ubuntu from the Microsoft Store.

2. Launch Ubuntu and follow the prompts to finish the setup.

### Run Automated Setup Script

```bash
curl -fsSLO https://raw.githubusercontent.com/VoltaML/voltaML-fast-stable-diffusion/experimental/scripts/wsl-install.sh
chmod +x wsl-install.sh
. wsl-install.sh
```

## How to start the application after closing it

::: tip
I have an easier way planned for the future, but for now, you'll have to run the following commands every time you want to start the application.
:::

1. Launch Ubuntu from the Start Menu
2. `cd voltaML-fast-stable-diffusion` (or wherever you cloned the repository)
3. `source venv/bin/activate`
4. `python main.py`
