import functools
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from diffusers import UNet2DConditionModel

if TYPE_CHECKING:
    from core.types import PyTorchModelType


logger = logging.getLogger(__name__)


def generate_traced_model(
    model: str,
    n_experiments: int = 2,
    unet_runs_per_experiment: int = 50,
    auth_token: Optional[str] = os.environ["HUGGINGFACE_TOKEN"],
):
    "Generate a traced model for the given model id"

    torch.set_grad_enabled(False)

    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
        StableDiffusionPipeline,
    )

    # load inputs
    def generate_inputs():
        sample = torch.randn(2, 4, 64, 64).half().cuda()
        timestep = torch.rand(1).half().cuda() * 999
        encoder_hidden_states = torch.randn(2, 77, 768).half().cuda()
        return sample, timestep, encoder_hidden_states

    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16,
        use_auth_token=auth_token,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
    )
    assert isinstance(pipe, StableDiffusionPipeline)
    pipe = pipe.to("cuda")
    unet = pipe.unet  # type: ignore
    unet.eval()
    unet.forward = functools.partial(unet.forward, return_dict=False)  # type: ignore

    # warmup
    for _ in range(3):
        with torch.inference_mode():
            inputs = generate_inputs()
            _ = unet(*inputs)

    # trace
    print("tracing..")
    unet_traced = torch.jit.trace(unet, inputs)  # type: ignore
    unet_traced.eval()  # type: ignore
    print("done tracing")

    # warmup and optimize graph
    for _ in range(5):
        with torch.inference_mode():
            inputs = generate_inputs()
            _ = unet_traced(*inputs)  # type: ignore

    # benchmarking
    with torch.inference_mode():
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                _ = unet_traced(*inputs)  # type: ignore
            torch.cuda.synchronize()
            print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                _ = unet(*inputs)  # type: ignore
            torch.cuda.synchronize()
            print(f"unet inference took {time.time() - start_time:.2f} seconds")

    # save the model
    author, model_name = model.split("/")

    path = Path(f"traced_unet/{author}")
    path.mkdir(exist_ok=True)
    unet_traced.save(f"traced_unet/{author}/{model_name}.pt")  # type: ignore


@dataclass
class UNet2DConditionOutput:
    "Output of the UNet2DCondition model"

    sample: torch.FloatTensor


class TracedUNet(torch.nn.Module):
    "UNet model optimized with JIT tracing"

    def __init__(self, unet_traced, untraced_unet: UNet2DConditionModel):
        super().__init__()
        self.in_channels = untraced_unet.in_channels
        self.device = untraced_unet.device
        self.unet_traced = unet_traced

    def forward(self, latent_model_input, t, encoder_hidden_states):
        "Forward pass of the UNet model"

        sample = self.unet_traced(latent_model_input, t, encoder_hidden_states)[0]
        return UNet2DConditionOutput(sample=sample)


def get_traced_unet(
    model_id: str, untraced_unet: UNet2DConditionModel
) -> Optional[TracedUNet]:
    "Get a traced UNet model"

    author, model_name = model_id.split("/")
    path = Path(f"traced_unet/{author}/{model_name}.pt")

    if path.exists():
        unet_traced = torch.jit.load(str(path))  # type: ignore
        unet_traced.eval()  # type: ignore
        unet = TracedUNet(unet_traced, untraced_unet)
        unet.to(dtype=torch.float16, device=untraced_unet.device)  # type: ignore
        return unet
    else:
        return None
