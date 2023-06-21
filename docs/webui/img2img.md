# Image to Image

This page is focused on transforming input image with the power of Stable Diffusion Models.
There are multiple ways to do this and we will cover all of the available options here.

If the parameter is not explained here, it is explained in the [Text to Image](/webui/txt2img) page.

::: warning
All of these models require extra model to be loaded and sometimes even one more for the detection algorithm. TLDR: It sucks a lot of VRAM.
:::

## Image to Image

This is the simplest way to transform an image. Stable Diffusion will take this image as an initial guide.

#### Denoising strength

The higher the value, the more of the image will be forgotten and transformed by the model.

::: info TODO
Plot of images showing the effect of this parameter
:::

## ControlNet

ControlNet is a neural network structure to control diffusion models by adding extra conditions. More information can be found in the [paper](https://arxiv.org/abs/2302.05543) or on [GitHub](https://github.com/lllyasviel/ControlNet).

For now, we only support 4 modes:

### Canny

::: info TODO
Show an example of the Canny mode (input, output, and the Canny edges)
:::

Canny is just an edge detection algorithm. It will detect edges in the image and use them as a guide for the model.
This approach doesn't require any additional models like OpenPose or MLSD, so it can be considered lightweight.

#### Low threshold and High threshold

These parameters are used by the Canny algorithm to detect edges. More broader values will detect more edges, but also more noise.

### HED

::: info TODO
Show an example of the HED mode (input, output, and the HED edges)
:::

More fancier edge detection algorithm. It requires extra model to be loaded, but is relatively lightweight.

### MLSD

::: info TODO
Show an example of the MLSD mode (input, output, and the MLSD edges)
:::

### OpenPose

::: info TODO
Show an example of the OpenPose mode (input, output, and the OpenPose edges)
:::

OpenPose is a pose estimation algorithm. It will detect human poses in the image and use them as a guide for the model.
It is heavier than the previous modes.
