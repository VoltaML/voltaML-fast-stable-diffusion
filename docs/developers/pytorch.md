# PyTorch

This is a guide to running this project with PyTorch only configuration.

## Requirements

- **Operating system:** Windows or Linux
- **Graphics card:** NVIDIA GPU with CUDA support
- **Driver version:** 515+

If you are running on Linux, you will need to install CUDA by following the instructions [here](https://developer.nvidia.com/cuda-11-7-0-download-archive).

## Running locally

### 1. Clone the repository

```bash
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion.git --branch experimental
```

### 2. Move into the project directory

```bash
cd voltaML-fast-stable-diffusion
```

### 3. Set up environmental variables

<br>

#### Windows

Please read [this guide](https://www.architectryan.com/2018/08/31/how-to-change-environment-variables-on-windows-10/) to learn how to set up environmental variables on Windows.

Variables that are stored there are persistent and will be available after restarting your computer.

#### Linux

```bash
export HUGGINGFACE_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

::: tip
You can also add the following line to your `~/.bashrc` file to make the variable persistent.
:::

### 4. Run the `main.py` file

::: warning
If you are using Linux, you might need to install `python3-virtualenv` package.
<br><br>
`sudo apt install python3-virtualenv`

For Windows users, run this command:
`pip install virtualenv`
:::
::: warning
If you are running Linux, you might need to use `python3` instead of `python`.
:::

```bash
python main.py
```

::: tip
If you are debugging the code, you can use the `--log-level=DEBUG` flag to see more detailed logs.
:::

### 5. Activate Virtual environment

<br>

#### Windows

```powershell
.\venv\Scripts\activate.ps1

or

.\venv\Scripts\activate.bat
```

#### Linux

```bash
source venv/bin/activate
```

### 6. Rerun the `main.py` file (it will install dependencies automatically)

```bash
python main.py
```

### 7. Access the API documentation to see if everything is working

You should now see that the WebUI is running on `http://localhost:5003/`.
There is an interactive documentation for the API available at `http://localhost:5003/api/docs`.
