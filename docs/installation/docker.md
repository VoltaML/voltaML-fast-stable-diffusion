# Docker

::: warning
This setup will require WSL2 on Windows. If you installed WSL or Docker before it was released, you will need to switch manually to WSL2.
:::

This is the easiest setup possible that should just work. Docker provides reproducible environments and makes it easy to run the application on any system.

## Requirements

- **Operating system:** Windows or Linux
- **Graphics card:** NVIDIA GPU with CUDA support
- **Driver version:** 515+ with CUDA
- **Docker**: [Docker Desktop](https://www.docker.com/products/docker-desktop)

## Main Branch (default - stable)

### 1. Clone the repository

```bash
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion --single-branch
```

### 2. Get inside the directory

```bash
cd voltaML-fast-stable-diffusion
```

### 3. Edit the `docker-compose.yml` file

```yaml
..
volumes:
  - XXX:/app/data
  - YYY/.cache/huggingface:/root/.cache/huggingface
..
```

where `XXX` is the path to the directory where you want to store all the data (converted models, outputs) and `YYY` is the path to your home directory (`C:\Users\YOUR_USERNAME` or `/home/USER`).

### 4. Edit the `.env` file

```bash
HUGGINGFACE_TOKEN=PLACE_YOUR_TOKEN_HERE
LOG_LEVEL=INFO # INFO, DEBUG, WARNING, ERROR
```

### 5. Run the container

```bash
docker compose run --service-ports voltaml
```

or if you have older version of docker

```bash
docker-compose run -p 5003:5003 voltaml
```

## Experimental Branch (development - unstable)

If you would like to have the latest features and bug fixes, and you do not mind having unstable container, then feel free to use this branch.

### 1. Clone the repository

```bash
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion -b experimental --single-branch
```

### 2. Get inside the directory

```bash
cd voltaML-fast-stable-diffusion
```

### 3. Edit the `docker-compose.yml` file

```yaml
..
image: stax124/volta:experimental
..
volumes:
  - XXX:/app/data
  - YYY/.cache/huggingface:/root/.cache/huggingface
..
```

Where `XXX` is the path to the directory where you want to store all the data (converted models, outputs) and `YYY` is the path to your home directory (`C:\Users\YOUR_USERNAME` or `/home/USER`).

Make sure you replace the `image` property with `stax124/volta:experimental`

### 4. Edit the `.env` file

```bash
HUGGINGFACE_TOKEN=PLACE_YOUR_TOKEN_HERE
LOG_LEVEL=INFO # INFO, DEBUG, WARNING, ERROR
```

### 5. Run the container

```bash
docker compose run --service-ports voltaml
```

or if you have older version of docker

```bash
docker-compose run -p 5003:5003 voltaml
```
