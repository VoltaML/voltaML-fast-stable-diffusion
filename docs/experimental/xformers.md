# xFormers

Xformers library is an optional way to speedup your image generation.

As of now, xFormers isn't available on standard PyPI, but there is a pre-release version available.

## Installation

::: tip
Docker images already include xFormers. There is no need to install it manually.
:::

Users need to install manually if they use local installation of volta.

```bash
pip install -U --pre xformers
```

## Usage

Volta should automatically detect if xFormers is installed and it will automatically apply the optimizations.
If that isn't the case, it will attempt to use SDP Attention, that is build into PyTorch 2.0.
