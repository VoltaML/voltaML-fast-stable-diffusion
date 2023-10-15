from typing import Any, Callable, Dict, Literal, Tuple, Union

from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser

Sampler = Tuple[str, Union[Callable, str], Dict[str, Any]]
Denoiser = Union[CompVisVDenoiser, CompVisDenoiser]

# UniPC
AlgorithmType = Literal["noise_prediction", "data_prediction"]
ModelType = Literal["v", "noise"]  # "v" for 2.x and "noise" for 1.x
Variant = Literal["bh1", "bh2"]
SkipType = Literal["logSNR", "time_uniform", "time_quadratic"]
Method = Literal["multistep", "singlestep", "singlestep_fixed"]
UniPCSchedule = Literal["discrete", "linear", "cosine"]
