# Welcome to VoltaML

<h2 align="center" style="border-bottom: 1px solid var(--vp-c-divider); padding-bottom: 24px;">
Made with ❤️ by <a href="https://github.com/Stax124" target="_blank">Stax124</a>
</h2>

::: info
Documentation is still a work in progress, if you have any questions, feel free to join our [Discord server](https://discord.gg/pY5SVyHmWm) or open an issue on GitHub.
:::

Stable Diffusion WebUI accelerated by <a href="https://github.com/facebookincubator/AITemplate">AITemplate</a>

**This documentation should walk you through the installation process, your first generated image, setting up the project to your liking and accelerating models with AITemplate.**

There is also a dedicated section to the **Discord bot, API** and a section for **developers and collaborators.**

## Main features

- Easy install with Docker
- Clean and simple Web UI
- Supports PyTorch as well as AITemplat for inference
- Support for Windows and Linux
- xFormers supported out of the box
- GPU cluster load balancing
- Discord bot
- Documented API
- Clean source code that should be easy to understand

## Feature availability

- ✅ Feature available and supported
- ❌ Feature not available yet
- 🚧 Feature is in the development or testing phase

| Feature          | PyTorch | AITemplate | Long Weighted Prompt (PyTorch Only) |
| ---------------- | ------- | ---------- | ----------------------------------- |
| Txt2Img          | ✅      | ✅         | ✅                                  |
| Img2Img          | ✅      | 🚧         | ✅                                  |
| ControlNet       | ✅      | ❌         | ❌                                  |
| Inpainting       | ✅      | 🚧         | ✅                                  |
| Image Variations | ❌      | ❌         | ❌                                  |
| SD Upscale       | ❌      | ❌         | ❌                                  |
| Depth2Img        | ❌      | ❌         | ❌                                  |
| Pix2Pix          | ❌      | ❌         | ❌                                  |

| Feature                   | Availability |
| ------------------------- | ------------ |
| Discord bot               | ✅           |
| Real-ESRGAN               | ❌           |
| Latent Upscale            | ❌           |
| Documentation             | ✅           |
| Image Browser             | ✅           |
| Model Conversion          | 🚧           |
| Model Training            | ❌           |
| Confiruration             | 🚧           |
| Multi-GPU                 | ✅           |
| MultiModel API            | ✅           |
| MultiModel UI             | ❌           |
| UI Performance monitoring | ✅           |

## Speed comparison

The below benchmarks have been done for generating a 512x512 image, batch size of one, measured in it/s.

| GPU         | PyTorch | xFormers | AITemplate |
| ----------- | ------- | -------- | ---------- |
| RTX 4090    | 19      | 39       | 60         |
| RTX 4080    | 15.53   | 20.21    | 40.51      |
| RTX 2080 Ti | 8       | No data  | No data    |
| RTX 3050    | 4.6     | 5.7      | 10.15      |
| RTX 3060 Ti | No data | 10.50    | 19.46      |
| A100        | 15.1    | 27.5     | No data    |
| A10         | 8.8     | 15.6     | 23.5       |
| T4          | 4.3     | 5.5      | No data    |

## UI Preview

**Text to image**
![Text2Image](/static/frontend/frontend-txt2img.webp)

<hr>

**Image to image**
![Image2Image](static/frontend/frontend-img2img.webp)

<hr>

**Image Browser**
![ImageBrowser](/static/frontend/frontend-browser.webp)
