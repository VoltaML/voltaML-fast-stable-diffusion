from typing_extensions import TYPE_CHECKING
from importlib.util import find_spec

if TYPE_CHECKING:
    from sfast.compilers.stable_diffusion_pipeline_compiler import CompilationConfig

from core.config import config as conf


def create_config() -> "CompilationConfig.Default":
    from sfast.compilers.stable_diffusion_pipeline_compiler import CompilationConfig

    config = CompilationConfig.Default()

    config.enable_xformers = (
        conf.api.sfast_xformers and find_spec("xformers") is not None
    )
    config.enable_triton = conf.api.sfast_triton and find_spec("triton") is not None
    config.enable_cuda_graph = conf.api.sfast_cuda_graph
    return config


def compile(model):
    if conf.api.sfast_compile:
        if hasattr(model, "sfast_compiled"):
            return model
        import sfast.jit.trace_helper
        from sfast.compilers.stable_diffusion_pipeline_compiler import compile as comp

        # stable-fast hijack since it seems to be break cuda graphs for now.
        # functionality-wise should be the same, except that it skips UNet2DConditionalOutput
        class BetterDictToDataClassConverter:
            def __init__(self, clz):
                self.clz = clz

            def __call__(self, d):
                try:
                    return self.clz(**d)
                except TypeError:
                    return d

        sfast.jit.trace_helper.DictToDataClassConverter = BetterDictToDataClassConverter

        r = comp(model, create_config())
        setattr(r, "sfast_compiled", True)
        return r
    return model
