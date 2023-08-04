from dataclasses import dataclass, field
from typing import List, Literal, Union

import torch


@dataclass
class APIConfig:
    "Configuration for the API"

    # Autoload
    autoloaded_textual_inversions: List[str] = field(default_factory=list)

    # Websockets and intervals
    websocket_sync_interval: float = 0.02
    websocket_perf_interval: float = 1.0

    # TomeSD
    use_tomesd: bool = False  # really extreme, probably will have to wait around until tome improves a bit
    tomesd_ratio: float = 0.25  # had to tone this down, 0.4 is too big of a context loss even on short prompts
    tomesd_downsample_layers: Literal[1, 2, 4, 8] = 1

    image_preview_delay: float = 2.0

    # General optimizations
    autocast: bool = False
    attention_processor: Literal[
        "xformers", "sdpa", "cross-attention", "subquadratic", "multihead"
    ] = "sdpa"
    subquadratic_size: int = 512
    attention_slicing: Union[int, Literal["auto", "disabled"]] = "disabled"
    channels_last: bool = True
    vae_slicing: bool = True
    vae_tiling: bool = False
    trace_model: bool = False
    clear_memory_policy: Literal["always", "after_disconnect", "never"] = "always"
    offload: bool = False
    data_type: Literal["float32", "float16", "bfloat16"] = "float16"

    # CUDA specific optimizations
    reduced_precision: bool = False
    cudnn_benchmark: bool = False
    deterministic_generation: bool = False

    # Device settings
    device_id: int = 0
    device_type: Literal["cpu", "cuda", "mps", "directml", "intel", "vulkan"] = "cuda"

    # Critical
    enable_shutdown: bool = True

    # VAE
    upcast_vae: bool = False

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

    @property
    def dtype(self):
        "Return selected data type"
        if self.data_type == "bfloat16":
            return torch.bfloat16
        if self.data_type == "float16":
            return torch.float16
        return torch.float32

    @property
    def device(self):
        "Return the device"

        if self.device_type == "intel":
            from core.inference.functions import is_ipex_available

            return torch.device("xpu" if is_ipex_available() else "cpu")

        if self.device_type in ["cpu", "mps"]:
            return torch.device(self.device_type)

        if self.device_type in ["vulkan", "cuda"]:
            return torch.device(f"{self.device_type}:{self.device_id}")

        if self.device_type == "directml":
            import torch_directml  # pylint: disable=import-error

            return torch_directml.device()
        else:
            raise ValueError(f"Device type {self.device_type} not supported")
