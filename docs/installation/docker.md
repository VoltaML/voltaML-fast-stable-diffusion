# Docker

If you are using Windows, refer to the [Local Installation guide](/installation/local#windows) as it provides a better experience.

::: warning
This setup will require WSL2 on Windows. If you installed WSL or Docker before it was released, you will need to switch manually to WSL2.
:::

This is the easiest setup possible that should just work. Docker provides reproducible environments and makes it easy to run the application on any system.

## Requirements

- **Operating system:** Windows or Linux
- **Graphics card:** NVIDIA GPU with CUDA support
- **Graphics card for AITemplate:** RTX 40xx, RTX 30xx, H100, A100, A10, A30, V100, T4
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
version: "3.7"

services:
  voltaml:
    image: stax124/volta:latest
    pull_policy: always // [!code focus]
    environment:
      - HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
      - LOG_LEVEL=${LOG_LEVEL}
      - FASTAPI_ANALYTICS_KEY=${FASTAPI_ANALYTICS_KEY}
      - DISCORD_BOT_TOKEN=${DISCORD_BOT_TOKEN}
      - EXTRA_ARGS=${EXTRA_ARGS}
    volumes: // [!code focus]
      - XXX:/app/data # XXX is the path to the folder where all the outputs will be saved // [!code focus]
      - YYY/.cache/huggingface:/root/.cache/huggingface # YYY is path to your home folder (you may need to change the YYY/. cache/huggingface to YYY\.cache\huggingface on Windows) // [!code focus]
    ports:
      - "5003:5003"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]

volumes:
  cache: {}
```

- `XXX` is the path to the directory where you want to store all the data (converted models, outputs)
- `YYY` is the path to your home directory (`C:\Users\YOUR_USERNAME` or `/home/USER`).
- `pull_policy: always` means that you will always try to get the latest image. Feel free to set it to `missing` to download only if it isn't present on your system

### 4. Edit the `.env` file

```bash
# Hugging Face Token (https://huggingface.co/settings/tokens)
HUGGINGFACE_TOKEN=YOUR_TOKEN_HERE // [!code focus]

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO

# [Optional] Analytics (https://my-api-analytics.vercel.app/generate) (https://my-api-analytics.vercel.app/dashboard)
FASTAPI_ANALYTICS_KEY=

# [Optional] Discord Bot Token (https://discord.com/developers/applications)
DISCORD_BOT_TOKEN=

# [Optional] Extra arguments for the API
EXTRA_ARGS=
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
