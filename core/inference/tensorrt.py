import logging

from core.tensorrt.TensorRT.engine import EngineBuilder
from core.types import BuildRequest, Job

from .base_model import InferenceModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TensorRTModel(InferenceModel):
    "High level wrapper for the TensorRT model"

    def __init__(self, model_id: str, use_f32: bool = False, device: str = "cuda"):
        self.model_id = model_id
        self.use_f32 = use_f32
        self.device = device

    def load(self):
        "Loads the model into the memory"

    def unload(self):
        "Unloads the model from the memory"

    def generate(self, job: Job):
        "Generates the output for the given job"

    def generate_engine(self, request: BuildRequest):
        "Generates a TensorRT engine from a local model"

        builder = EngineBuilder(
            model_id=request.model_id,
            hf_token=request.hf_token,
            fp16=request.fp16,
            verbose=request.verbose,
            opt_image_height=request.opt_image_height,
            opt_image_width=request.opt_image_width,
            max_batch_size=request.max_batch_size,
            onnx_opset=request.onnx_opset,
            build_static_batch=request.build_static_batch,
            build_dynamic_shape=request.build_dynamic_shape,
            build_preview_features=request.build_preview_features,
            force_engine_build=request.force_engine_build,
            force_onnx_export=request.force_onnx_export,
            force_onnx_optimize=request.force_onnx_optimize,
            onnx_minimal_optimization=request.onnx_minimal_optimization,
        )

        print(builder)
        builder.build()
        print("Builder finished")
