from abc import ABC, abstractmethod
from typing import Any, Optional


class PipelineMixin(ABC):
    """
    Since the diffusion procedure itself remains hugely unchanged, I think it'd
    make development more "streamlined" if we were to merge all of them into one.
    This can, and more importantly, WILL introduce SOME weird issues when using
    different architectures compared to just Stable Diffusion. (or PyTorch, for
    that matter) This pipeline should also, in theory, support models that use:

    a) multiple text encoders

        a. multiple, as in, more than one text encoder, for instance, in SDXL

        b. multiple, as in, multiple different architectures, like T5 for IF

    b) prior transformers, emb2emb

    c) unCLIP support

    d) different schedulers - some k-diffusion ones, some only diffusers

    e) different backends - onnx/ait/tensorrt/pytorch, whatever we throw at it,
    it should function.

    f) watermarking - for models that need it; i.e.: IF, SDXL (maybe toggleable)
    """

    @property
    @abstractmethod
    def multiples_of(self) -> int:
        "!!! (size % self.multiples_of == 0) !!!"

    @abstractmethod
    def load_modifications(self, *args, **kwargs) -> None:
        "Load modifications"

    @abstractmethod
    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        images_per_prompt: int = 1,
    ) -> Any:
        "Encode prompt"
