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

Volta should automatically detect if xFormers is installed and it will automatically apply the optimizations.
If that isn't the case, it will attempt to use SDP Attention, that is build into PyTorch 2.0.
