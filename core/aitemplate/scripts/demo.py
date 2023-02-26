#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import torch
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from diffusers import EulerDiscreteScheduler

from ..src.pipeline_stable_diffusion_ait import StableDiffusionAITPipeline


def run(
    local_dir: str,
    prompt: str,
    width: int = 512,
    height: int = 512,
    benchmark: bool = False,
):
    pipe = StableDiffusionAITPipeline.from_pretrained(
        local_dir,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            local_dir, subfolder="scheduler"
        ),
        revision="fp16",
        torch_dtype=torch.float16,
    )
    assert isinstance(pipe, StableDiffusionAITPipeline)
    pipe.to("cuda")

    with torch.autocast("cuda"):  # type: ignore
        image = pipe(prompt, height, width).images[0]  # type: ignore
        if benchmark:
            t = benchmark_torch_function(10, pipe, prompt, height=height, width=width)
            print(f"sd e2e: {t} ms")

    image.save("example_ait.png")
