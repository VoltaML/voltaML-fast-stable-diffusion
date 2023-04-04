# Checkpoints

This document describes the current state of support for .ckpt or .safetensors files.

## Upsides

- Allows you to load a checkpoint from sites like [CivitAI](https://civit.ai/).
- You can use checkpoints from A111
- Files can be easily transported between machines

## Downsides

- Slow to load
- Will for now require downloading of the Safety Checker model as well - [maybe fixed in future](https://github.com/huggingface/diffusers/pull/2768)
- Only for PyTorch inference for now, AITemplate support **might** be added in the future

## How to use

All models that are stored in the `data/models` should be available for direct use. You can select these models like any other diffusers model in the UI.

We support both `.ckpt` and `.safetensors` files. The `.ckpt` files are the raw checkpoint files that you can download from sites like [CivitAI](https://civit.ai/). The `.safetensors` files are the same files but stored in a more safe way (ckpt can have malicious code in it).
