# Changelog

## v.0.6.0

### All changes

- SDXL support
- Removed original bislerp
- FlashAttention
- Better clip skip
- Automatic model version detection
- SAG rework
- Updated default settings for lower VRAM usage
- Settings difference resolver (migration to newer versions should be smoother)
- Partial support for float8 (only for storing models in VRAM, but drastically reduces VRAM usage)
- Heun++ sampler
- Kohya deepshrink
- Scalecrafter
- Added CODEOWNERS
- Updated CivitAI browser UI
- SASolver
- General postprocessors (hi-res, upscales, etc.)
- UI improvements for mobile users (WIP)
- FreeU v2 support, default FreeU v2 settings picker
- ADetailer
- More predictable model loading

## v0.5.0

- Highres fix can now use image upscalers (ESRGAN, RealSR, etc.) for the intermediate step
- API converted to sync where applicable, this should resolve some issues with websockets and thread lockups
- FreeU support
- Diffusers version bump

## v0.4.2

### Critical fix

- PyTorch will now download CUDA version instead of CPU version if available

## v0.4.1

### Bug Fixes

- Fixed loras on latest version of diffusers
- Fixed Karras sigmas
- Fixed incorrect step count being displayed in the UI
- Fixed CivitAI browser getting stuck in some scenarios

### All changes

- Added support for prompt expansion
- Reorganized frontend code

## v0.4.0

### Biggest Changes

- Hi-res fix for AITemplate
- Model and VAE autoloading
- Partial support for Kdiffusion samplers (might be broken in some cases - controlnet, hi-res...)
- Hypertile support (https://github.com/tfernd/HyperTile)
- Theme overhaul

### All Changes

- Added docs for Vast.ai
- Better Settings UI
- Updated Model Manager
- Garbage Collection improvements
- Hi-res fix for AITemplate
- Added DPMSolverSDE diffusers sampler
- Model autoloading
- Partial support for Kdiffusion samplers (might be broken in some cases - controlnet, hi-res...)
- Tag autofill
- New SendTo UI that is context aware
- Fixed symlink deletion bug for models
- New documentation theme
- New sigma types for Kdiffusion (exponential, polyexponential, VP)
- Image upload should now display correct dimensions
- Fixed WebP crashes in some cases
- Remote LoRA support `<lora:URL:weight>`
- Fix some CORS issues
- Hypertile support (https://github.com/tfernd/HyperTile)
- Fixed uvicorn logging issues
- Fixed update checker
- Added some extra tooltips to the UI
- Sampler config override for people that hate their free time
- Bumped dependency versions
- Image browser entries should get sorted on server, removing the need for layout shift in the UI
- Cleaned up some old documentation
- Transfer project from PyLint to Ruff
- Github Actions CI for Ruff linting
- Theme overhaul
- Fixed NaiveUI ThemeEditor
- Sort models in Model Loader
- Console logs now accessible in the UI
- ...and probably a lot more that I already forgot

### Contributors

- gabe56f (https://github.com/gabe56f)
- Stax124 (https://github.com/Stax124)
- Katehuuh (https://github.com/Katehuuh)

### Additional Notes

Thank you for 850 stars on GitHub and 500 Discord members ❤️
