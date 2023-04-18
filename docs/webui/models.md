# Models

VoltaML support all of these methods for loading models:

|                    | Diffusers | Checkpoint (.ckpt)     | Safetensors (.safetensors) | AITemplate  |
| ------------------ | --------- | ---------------------- | -------------------------- | ----------- |
| AITemplate compile | Yes       | No                     | No                         | Unavailable |
| Float 16           | Yes       | Only if already pruned | Only if already pruned     | Yes         |
| Float 32           | Yes       | Yes                    | Yes                        | No          |

Both `Checkpoint` and `Safetensors` are loaded with float type that they were saved with. Fix needs to be done in the `diffusers` package - follow [this](https://github.com/huggingface/diffusers/issues/2755) thread for more info.

## AITemplate compilation

The AITemplate compilation is a process that traces the model and creates more optimized version of it. This process can be started on the `Accelerate` page.

## Float16 / Float32

This refers to the precision of the model. Float16 is a half precision model, which is faster to load and run, but less accurate. Float32 is a full precision model, which is slower to load and run, but more accurate.

It can be also seen on the filesizes of the models. Float16 models are 2x smaller than Float32 models (2GB compared to 4GB).

## Model conversion

Work in progress.
