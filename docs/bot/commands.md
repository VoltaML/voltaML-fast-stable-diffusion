# Commands

This is a list of commands that the bot supports. The bot will also try respond to any command prefixed with `!`, but the intended way of using these commands is with a `/`.

## Available

Lists all the available models that can be loaded for inference.

## Clean-memory

Manually clean the VRAM. For debugging purposes.

## Dream

Generate an image from a prompt.

- `prompt` - The prompt to generate the image from.
- `model` - The model to use.
- `negative_prompt` - (Optional) The negative prompt to generate the image from. Default: `See source code - its too big for this page`.
- `guidance_scale` - (Optional) How closely the model should follow the prompt. Lower values will result in more creative images. Higher values will result in more predictable images. Default: `7`.
- `steps` - (Optional) How many steps to take in the inference process. Default: `30`.
- `resolution` - (Optional) The resolution of the generated image. Options: `512x512`,`1024x1024`,`512x912`,`912x512`,`1920x1080`,`1080x1920`,`1280x720`, `720x1280`,`768x768`. Default: `512x512`.
- `seed` - (Optional) The seed to use for the inference process. Default: `Random seed generated on runtime`.
- `scheduler` - (Optional) The scheduler to use for the inference process. Default: `13` (UniPCMultistep).
- `use_default_negative_prompt` - (Optional) Whether to use the default negative prompt. Default: `True`.
- `verbose` - (Optional) Whether to show generation table with all the information. Default: `False`.

## GPUS

List all the GPUs available that the API can use.

## Load

Load a locally saved model.

- `model` - The model to load.
- `device` - (Optional) The device to use for the inference process. Default: `cuda`.
- `backend` - (Optional) The backend to use for the inference process. Options: `PyTorch`,`TensorRT`. Default: `PyTorch`.

## Loaded

List all the loaded models.

## Reset-queue

Clears the queue in case a job gets stuck. For debugging purposes.

## Sync

::: info
This command is for developers and debugging. You need to run this command only if you updated the command parameters and you want to sync them with Discord.
:::

Sync all the commands with Discord. This is only needed the first time you start the bot and if you followed the docs properly, you already have this set up correctly.

## Unload

Unload a locally saved model.
