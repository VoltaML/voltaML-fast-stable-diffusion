import json
from pathlib import Path


def get_unet_in_channels(directory: Path):
    "Get the number of input channels for the UNet model."

    try:
        with directory.joinpath("config.json").open() as f:
            config = json.load(f)
            return int(config["unet_in_channels"])
    except FileNotFoundError:
        return 4
