# xFormers

Xformers library is an optional way to save some of your VRAM.

## Downsides

Images are no longer determinisic. This means that you can't use the same seed to get the same results.

## Installation

Users need to install manually if they use local installation of volta.

```bash
pip install xformers
```

## Usage

Head over to `Settings > API` and select `xFormers` in `Attention processor`.
You need to reload the model after changing this setting to apply the changes.
