# Welcome to VoltaML

::: info
Documentation is still a work in progress, if you have any questions, feel free to join our [Discord server](https://discord.gg/pY5SVyHmWm) or open an issue on GitHub.
:::

Stable Diffusion WebUI and API accelerated by <a href="https://developer.nvidia.com/tensorrt">TensorRT</a>

**This documentation should walk you through the installation process, your first generated image, setting up the project to your liking and accelerating models with TensorRT.**

**There is also a dedicated section to the Discord bot, API and a section for **developers and collaborators.\*\*\*\*

## Main features

- Easy install with Docker
- Clean and simple Web UI
- Supports PyTorch as well as TensorRT for the fastest inference
- Support for Windows and Linux (TRT is not officially supported on Windows if running locally)
- xFormers support
- GPU cluster support with load balancing
- Discord bot

## Speed comparison

::: warning
Old data, in need of rerun - observed speedup should be approximately 2.5x
:::

The below benchmarks have been done for generating a 512x512 image, batch size of one (measured in it/s).

| GPU         | PyTorch | xFormers | TensorRT |
| ----------- | ------- | -------- | -------- |
| RTX 4090    | 19      | 40       | 87       |
| RTX 2080 Ti | 8       | No data  | 26.2     |
| RTX 3050    | 4.6     | 5.7      | 12.5     |
| A100        | 15.1    | 27.5     | 62.8     |
| A10         | 8.8     | 15.6     | 29.2     |
| T4          | 4.3     | 5.5      | 11.4     |

## UI Preview

**Text to image**
<img src="static/frontend/frontend-txt2img.webp" alt="screenshot" />

<hr>

**Image to image**
<img src="static/frontend/frontend-img2img.webp" alt="screenshot" />

<hr>

**Image Browser**
<img src="static/frontend/frontend-browser.webp" alt="screenshot" />
