from dataclasses import dataclass, field
from typing import Dict, List, Literal, Union

import torch


@dataclass
class APIConfig:
    "Configuration for the API"

    # Autoload
    autoloaded_textual_inversions: List[str] = field(default_factory=list)
    autoloaded_models: List[str] = field(default_factory=list)
    autoloaded_vae: Dict[str, str] = field(default_factory=dict)

    # Websockets and intervals
    websocket_sync_interval: float = 0.02
    websocket_perf_interval: float = 1.0
    enable_websocket_logging: bool = True

    # TomeSD
    use_tomesd: bool = False  # really extreme, probably will have to wait around until tome improves a bit
    tomesd_ratio: float = 0.25  # had to tone this down, 0.4 is too big of a context loss even on short prompts
    tomesd_downsample_layers: Literal[1, 2, 4, 8] = 1

    # General optimizations
    autocast: bool = False
    attention_processor: Literal[
        "xformers", "sdpa", "cross-attention", "subquadratic", "multihead"
    ] = "sdpa"
    subquadratic_size: int = 512
    attention_slicing: Union[int, Literal["auto", "disabled"]] = "disabled"
    channels_last: bool = True
    trace_model: bool = False
    clear_memory_policy: Literal["always", "after_disconnect", "never"] = "always"
    offload: Literal["disabled", "model", "module"] = "disabled"
    data_type: Literal["float32", "float16", "bfloat16"] = "float16"
    dont_merge_latents: bool = (
        False  # Will drop performance, but could help with some VRAM issues
    )

    # CUDA specific optimizations
    reduced_precision: bool = False
    cudnn_benchmark: bool = False
    deterministic_generation: bool = False

    # Device settings
    device: str = "cuda:0"

    # Critical
    enable_shutdown: bool = True

    # CLIP
    clip_skip: int = 1
    clip_quantization: Literal["full", "int8", "int4"] = "full"

    huggingface_style_parsing: bool = False

    # Saving
    save_path_template: str = "{folder}/{prompt}/{id}-{index}.{extension}"
    image_extension: Literal["png", "webp", "jpeg"] = "png"
    image_quality: int = 95
    image_return_format: Literal["bytes", "base64"] = "base64"

    # Grid
    disable_grid: bool = False

    # Torch compile
    torch_compile: bool = False
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: Literal[
        "default",
        "reduce-overhead",
        "max-autotune",
    ] = "reduce-overhead"

    sfast_compile: bool = False
    sfast_xformers: bool = True
    sfast_triton: bool = True
    sfast_cuda_graph: bool = True

    # Hypertile
    hypertile: bool = False
    hypertile_unet_chunk: int = 256

    # K_Diffusion
    sgm_noise_multiplier: bool = False  # also known as "alternate DDIM ODE"
    kdiffusers_quantization: bool = True  # improves sampling quality

    # K_Diffusion & Diffusers
    # What to do with refiner:
    #  - "joint:" instead of creating a new sampler, it uses the refiner inside of the main loop,
    #             replacing the unet with the refiners unet after a certain number of steps have
    #             been processed. This improves consistency and generation quality.
    #  - "separate:" creates a new pipeline for refiner and does the refining there on the final
    #                latents of the image. This can introduce some artifacts/lose context.
    sdxl_refiner: Literal["joint", "separate"] = "separate"

    # "philox" is what a "cuda" generator would be, except, it's on cpu
    generator: Literal["device", "cpu", "philox"] = "device"

    # VAE
    live_preview_method: Literal["disabled", "approximation", "taesd"] = "approximation"
    live_preview_delay: float = 2.0
    vae_slicing: bool = True
    vae_tiling: bool = True
    upcast_vae: bool = False  # Fixes issues on 10xx-series and RX cards
    # Somewhat fixes extraordinarily high CFG values. Does also change output composition, so
    # best to leave on off by default. TODO: write docs for this?
    apply_unsharp_mask: bool = False
    # Rescales CFG to a known good value when CFG is higher than this number. Set to "off" to disable.
    cfg_rescale_threshold: Union[float, Literal["off"]] = 10.0

    # Prompt expansion (very, and I mean VERYYYY heavily inspired/copied from lllyasviel/Fooocus)
    prompt_to_prompt: bool = False
    prompt_to_prompt_model: Literal[
        "lllyasviel/Fooocus-Expansion",
        "daspartho/prompt-extend",
        "succinctly/text2image-prompt-generator",
        "Gustavosta/MagicPrompt-Stable-Diffusion",
        "Ar4ikov/gpt2-medium-650k-stable-diffusion-prompt-generator",
    ] = "lllyasviel/Fooocus-Expansion"
    prompt_to_prompt_device: Literal["cpu", "gpu"] = "gpu"

    # Free U
    free_u: bool = False
    free_u_s1: float = 0.9
    free_u_s2: float = 0.2
    free_u_b1: float = 1.2
    free_u_b2: float = 1.4

    @property
    def dtype(self):
        "Return selected data type"
        if self.data_type == "bfloat16":
            return torch.bfloat16
        if self.data_type == "float16":
            return torch.float16
        return torch.float32

    @property
    def overwrite_generator(self) -> bool:
        "Whether the generator needs to be overwritten with 'cpu.'"

        return any(
            map(lambda x: x in self.device, ["mps", "directml", "vulkan", "intel"])
        )
