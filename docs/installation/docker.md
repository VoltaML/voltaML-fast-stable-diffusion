# Docker

## Requirements

- **Operating system:** Windows or Linux
- **Graphics card:** NVIDIA GPU with CUDA support
- **Driver version:** 515+ with CUDA
- **Docker**: [Docker Desktop](https://www.docker.com/products/docker-desktop)

### 1. Clone the repository

```bash
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion -b experimental --single-branch
```

### 2. Edit the `docker-compose.yml` file

```yaml
volumes:
  - XXX:/app/data
  - YYY/.cache/huggingface:/root/.cache/huggingface
```

where `XXX` is the path to the directory where you want to store the data and `YYY` is the path to your home directory (C:\Users\YOUR_USERNAME).

### 3. Edit the `.env` file

```bash
HUGGINGFACE_TOKEN=PLACE_YOUR_TOKEN_HERE
LOG_LEVEL=INFO # INFO, DEBUG, WARNING, ERROR
```

### 4. Run the container

```bash
docker compose run --service-ports voltaml
```

or if you have older version of docker

```bash
docker-compose run -p 5003:5003 voltaml
```
