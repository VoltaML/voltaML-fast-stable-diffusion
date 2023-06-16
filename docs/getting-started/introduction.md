# Welcome to VoltaML

<h2 align="center" style="border-bottom: 1px solid var(--vp-c-divider); padding-bottom: 24px;">
Made with ❤️ by <a href="https://github.com/Stax124" target="_blank">Stax124</a>
</h2>

::: danger IMPORTANT
For all Pull Requests, please make sure to target the `experimental` branch. The `main` branch is only used for releases (or in some situations, issues and PRs with high priority - marked as `Fast-Forward`).
:::

::: info
Documentation is still a work in progress, if you have any questions, feel free to join our [Discord server](https://discord.gg/pY5SVyHmWm) or open an issue on GitHub.
:::

Stable Diffusion WebUI accelerated by <a href="https://github.com/facebookincubator/AITemplate">AITemplate</a>

**This documentation should walk you through the installation process, your first generated image, setting up the project to your liking and accelerating models with AITemplate.**

There is also a dedicated section to the **Discord bot, API** and a section for **developers and collaborators.**

## Main features

- Easy install with Docker
- Clean and simple Web UI
- Supports PyTorch as well as AITemplate for inference
- Support for Windows and Linux
- xFormers supported out of the box
- GPU cluster load balancing
- Discord bot
- Documented API
- Clean source code that should be easy to understand

## Speed comparison

The below benchmarks have been done for generating a 512x512 image, batch size of one, measured in it/s.

| GPU             | PyTorch | SPDA  | AITemplate |
| --------------- | ------- | ----- | ---------- |
| RTX 4090        | 19      | 39    | 60         |
| RTX 4080        | 15.53   | 20.21 | 40.51      |
| RTX 3070 Laptop | No data | 9.8   | 16.8       |
| RTX 3050        | 4.6     | 5.7   | 10.15      |
| RTX 3060 Ti     | No data | 10.50 | 19.46      |
| A100            | 15.1    | 27.5  | No data    |
| A10             | 8.8     | 15.6  | 23.5       |
| T4              | 4.3     | 5.5   | No data    |

## UI Preview

**Text to image**
![Text2Image](../static/frontend/frontend-txt2img.webp)

<hr>

**Image to image**
![Image2Image](../static/frontend/frontend-img2img.webp)

<hr>

**Image Browser**
![ImageBrowser](../static/frontend/frontend-browser.webp)
