from dataclasses import dataclass, field
from typing import Dict, List, Literal, Union

import torch

from core.flags import LatentScaleModel


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
    attention_slicing: Union[int, Literal["auto", "disabled"]] = "auto"
    channels_last: bool = True
    trace_model: bool = False
    clear_memory_policy: Literal["always", "after_disconnect", "never"] = "always"
    offload: Literal["disabled", "model", "module"] = "disabled"
    data_type: Literal[
        "float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"
    ] = "float16"

    # VRAM optimizations
    # whether to run both parts of CFG>1 generations in one call. Increases VRAM usage during inference,
    # halves inference speed for most -- newer than 10xx -- cards.
    batch_cond_uncond: bool = True
    # Whether to cache the weight of tensors for LoRA loading during float8 inference.
    # Improves how LoRAs work when data-type is a subset of FP8, but increases system RAM usage to 2x.
    # ONLY WORKS WHEN DATA_TYPE=FLOAT8
    cache_fp16_weight: bool = False  # only works on float8. Used for LoRAs.

    # Approximation-type optimizations ("ruin" quality for big boosts in performance)
    # According to the following paper: https://arxiv.org/pdf/2312.09608.pdf (Faster-Diffusion)
    #   ControlNet infers can be "effectively skipped" after a certain point, because their control
    #   on the images loosens up after a while.
    #
    #   value: from which "percentage" of the diffusion process should we skip controlnet inferring.
    #   default: 1.0, don't skip at all.
    #   IMPORTANT: the first min(5, n-3) steps WILL always have controlnet inferring on to avoid completely
    #              disabling controlnet
    approximate_controlnet: float = 1.0

    # According to the following paper: https://browse.arxiv.org/html/2312.12487v1 (Adaptive Guidance)
    #   even stopping it naively (without implementing AG, which I (Gabe) might implement later)
    #   doesn't murder image quality. I'd compare it to how TensorRT "mutates" images.
    #   Probably shouldn't be set to anything below 0.75
    #
    #   value: from which "percentage" of the diffusion process should we stop inferring uncond.
    #   default: 1.0, don't stop inferring at all.
    cfg_uncond_tau: float = 1.0

    # Won't implement timestep pararellization since it has a tendency to "explode" VRAM usage, and I
    # don't know if I'm ready for issues that could bring down the line...
    # source: https://arxiv.org/pdf/2312.09608.pdf (Faster-Diffusion)

    # According to the following paper: https://arxiv.org/pdf/2312.09608.pdf (Faster-Diffusion)
    #   Dropping encode/decode -- effectively caching it -- and creating new values every nth step
    #   produces negligible quality loss whilst boosting performance by 1/drop_encode_decode%.
    #
    #   value: "off" disables this. "on" infers the first 5 steps ALWAYS as full-quality ones, and then every 5th.
    #   default: "off," due to quality loss, same reason as to why HyperTile is off by default.
    drop_encode_decode: Union[
        int,  # if int, drop every x-th step
        Literal["off", "on"],  # on = first 5 + every 5th
    ] = "on"  # "on" results in a ~30% increase in performance, without ruining quality too much

    deepcache_cache_interval: int = 1  # cache every x-th layer, 1 = disabled

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
    hypertile_unet_chunk: int = 512

    # Kohya Deep-Shrink
    deepshrink_enabled: bool = True
    deepshrink_depth_1: int = 3  # -1 to 12; steps of 1
    deepshrink_stop_at_1: float = 0.15  # 0 to 0.5; steps of 0.01

    deepshrink_depth_2: int = 4  # -1 to 12; steps of 1
    deepshrink_stop_at_2: float = 0.30  # 0 to 0.5; steps of 0.01

    deepshrink_scaler: LatentScaleModel = "bislerp"
    deepshrink_base_scale: float = 0.5  # 0.05 to 1.0; steps of 0.05
    deepshrink_early_out: bool = True

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
    live_preview_method: Literal[
        "disabled",
        "approximation",
        "taesd",
        "full",  # TODO: isn't supported yet.
    ] = "approximation"
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
    def dtype(self) -> torch.dtype:
        "Return selected data type"
        return getattr(torch, self.data_type)

    @property
    def load_dtype(self) -> torch.dtype:
        "Data type for loading models."
        dtype = self.dtype
        if "float8" in self.data_type:
            from core.shared_dependent import gpu

            if self.device == "cpu":
                if "bfloat16" in gpu.capabilities.supported_precisions_cpu:
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32
            else:
                if "float16" in gpu.capabilities.supported_precisions_gpu:
                    dtype = torch.float16
                else:
                    dtype = torch.float32
        return dtype

    @property
    def overwrite_generator(self) -> bool:
        "Whether the generator needs to be overwritten with 'cpu.'"

        return any(
            map(lambda x: x in self.device, ["mps", "directml", "vulkan", "intel"])
        )
