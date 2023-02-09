# Requirements

- **Operating system:** Windows or Linux
- **Graphics card:** NVIDIA GPU with CUDA support
- **Driver version:** 515+

# Windows

Only PyTorch is currently officially supported on Windows as Docker with WSL 2 has some problems with shared libraries.


If you are running on Linux, you will need to install CUDA by following the instructions [here](https://developer.nvidia.com/cuda-11-7-0-download-archive).

## 1. Clone the repository

```bash
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion.git --branch experimental
```

## 2. Move into the project directory

```bash
cd voltaML-fast-stable-diffusion
```

## 3. Set up environmental variables

Please read [this guide](https://www.architectryan.com/2018/08/31/how-to-change-environment-variables-on-windows-10/) to learn how to set up environmental variables on Windows.

Variables that are stored there are persistent and will be available after restarting your computer.

## 4. Virtual environment

Create a python virtual environment

:::info
If this is your first time using `virtualenv` then you might need to do `pip install virtualenv`
:::

```bash
virtualenv venv
```

And activate this environment with:

**Powershell**
```powershell
.\venv\Scripts\activate.ps1
```

**CMD**
```
.\venv\Scripts\activate.bat
```

**Bash (MINGW or GitBash)**
```bash
source venv/Scripts/activate
```

## 5. Run the main script

```
python main.py
```

All dependencies should get installed automatically

## 6. Open the WebUI

Please open your browser and navigate to `localhost:5003`