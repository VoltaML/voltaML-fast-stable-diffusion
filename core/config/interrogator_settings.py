from dataclasses import dataclass


@dataclass
class InterrogatorConfig:
    "Configuration for interrogation models"

    # set to "Salesforce/blip-image-captioning-base" for an extra gig of vram
    caption_model: str = "Salesforce/blip-image-captioning-large"
    visualizer_model: str = "ViT-L-14/openai"

    offload_captioner: bool = False
    offload_visualizer: bool = False

    chunk_size: int = 2048  # set to 1024 for lower vram usage
    flavor_intermediate_count: int = 2048  # set to 1024 for lower vram usage

    flamingo_model: str = "dhansmair/flamingo-mini"

    caption_max_length: int = 32
