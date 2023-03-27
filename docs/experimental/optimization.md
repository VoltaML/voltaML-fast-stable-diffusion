# Optimization

Volta supports multiple optimization levels for PyTorch. Optimization levels can be set in multiple ways:

1. `-o [0,1,2,3,4]` / `--optimization [0,1,2,3,4]` flag in the command line when starting volta.
2. `OPT_LEVEL` environment variable.
3. UI Settings page: `Settings > API > Optimization`

## Optimization Levels

Higher the number, the less VRAM it should consume - but with a performance hit.

- `0`: Traced UNet - Takes about 15s to trace (happens every load)
- `1`: Default - No extra stuff applied
- `2`: Splits the step into smaller pieces (saves some VRAM for high resolution images) - UNet Attention Slicing
- `3`: Splitting into smaller pieces and offloading model to CPU when not needed (for multimodel setup) - Offloaded VAE & UNet to CPU + UNet Slicing
- `4`: Offload unused componets to CPU (saves a lot of VRAM - **recommended for 4GB cards**) - Sequential Offload + UNet slicing
