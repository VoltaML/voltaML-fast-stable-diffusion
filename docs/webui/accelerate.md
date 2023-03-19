# Accelerate :rocket:

This page will teach you how to speed up your PyTorch model with **AITemplate**

## Upsides of using AITemplate

- about **1.9x** faster inference
- can be accelerated on a consumer GPU
- can technically be transferred to another device (within the same architecture)

## Drawbacks of using AITemplate

- more VRAM usage
- takes quite a bit of time to compile model
- only static sizes supported for now

## How to use

1. Set your static Width and Height (must be multiples of 32)
2. Set your batch size (probably leave at 1 if you're using a consumer GPU)
3. Select how many threads (jobs) you want to use - more threads means more RAM usage but faster compilation
4. Select the desired model - it needs to be downloaded first via the download page (may require page refresh to show up)
5. Click the `Accelerate` button
6. Wait for the model to compile (progress will be shown in `Acceleration progress` container)
