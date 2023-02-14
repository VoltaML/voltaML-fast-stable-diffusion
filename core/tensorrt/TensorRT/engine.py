# Original: https://github.com/ddPn08/Lsmith
# Modified by: Stax124

import gc
import json
import os

import torch

import onnx

from ..utilities import Engine
from .models import VAE, UNet


def get_model_path(name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, name + (".opt" if opt else "") + ".onnx")


def build_engine(
    model_name: str,
    engine_dir: str,
    onnx_dir: str,
    model_data,
    opt_image_height: int = 512,
    opt_image_width: int = 512,
    onnx_opset: int = 16,
    force_engine_build: bool = False,
    force_onnx_export: bool = False,
    force_onnx_optimize: bool = False,
    onnx_minimal_optimization: bool = False,
    build_static_batch: bool = False,
    build_dynamic_shape: bool = False,
    build_preview_features: bool = False,
):
    engine = Engine(model_name, engine_dir)
    if force_engine_build or not os.path.exists(engine.engine_path):
        onnx_path = get_model_path(model_name, onnx_dir, opt=False)
        onnx_opt_path = get_model_path(model_name, onnx_dir)
        if not os.path.exists(onnx_opt_path):
            if force_onnx_export or not os.path.exists(onnx_path):
                print(f"Exporting model: {onnx_path}")
                model = model_data.get_model()
                with torch.inference_mode(), torch.autocast("cuda"):  # type: ignore
                    inputs = model_data.get_sample_input(
                        1, opt_image_height, opt_image_width
                    )
                    torch.onnx.export(
                        model,
                        inputs,
                        onnx_path,
                        export_params=True,
                        opset_version=onnx_opset,
                        do_constant_folding=True,
                        input_names=model_data.get_input_names(),
                        output_names=model_data.get_output_names(),
                        dynamic_axes=model_data.get_dynamic_axes(),
                    )
                del model
                torch.cuda.empty_cache()
                gc.collect()
            else:
                print(f"Found cached model: {onnx_path}")
            if force_onnx_optimize or not os.path.exists(onnx_opt_path):
                print(f"Generating optimizing model: {onnx_opt_path}")
                onnx_opt_graph = model_data.optimize(
                    onnx.load(onnx_path),
                    minimal_optimization=onnx_minimal_optimization,
                )
                onnx.save(onnx_opt_graph, onnx_opt_path)
        engine.build(
            onnx_opt_path,
            fp16=True,
            input_profile=model_data.get_input_profile(
                1,
                opt_image_height,
                opt_image_width,
                static_batch=build_static_batch,
                static_shape=not build_dynamic_shape,
            ),
            enable_preview=build_preview_features,
        )

    return engine


class EngineBuilder:
    def __init__(
        self,
        model_id: str,
        hf_token="",
        fp16=False,
        verbose=False,
        opt_image_height=512,
        opt_image_width=512,
        max_batch_size=1,
        onnx_opset=16,
        build_static_batch=False,
        build_dynamic_shape=True,
        build_preview_features=False,
        force_engine_build=False,
        force_onnx_export=False,
        force_onnx_optimize=False,
        onnx_minimal_optimization=False,
    ):
        self.device = "cuda"
        self.hf_token = hf_token
        self.fp16 = fp16
        self.verbose = verbose
        self.opt_image_height = opt_image_height
        self.opt_image_width = opt_image_width
        self.onnx_opset = onnx_opset
        self.build_static_batch = build_static_batch
        self.build_dynamic_shape = build_dynamic_shape
        self.build_preview_features = build_preview_features
        self.force_engine_build = force_engine_build
        self.force_onnx_export = force_onnx_export
        self.force_onnx_optimize = force_onnx_optimize
        self.onnx_minimal_optimization = onnx_minimal_optimization
        self.models = {
            "unet": UNet(
                model_id,
                hf_token=hf_token,
                fp16=fp16,
                device=self.device,
                verbose=verbose,
                max_batch_size=max_batch_size,
            ),
            "vae": VAE(
                model_id,
                hf_token=hf_token,
                device=self.device,
                verbose=verbose,
                max_batch_size=max_batch_size,
            ),
        }

        self.model_dir = os.path.join(
            "engine",
            os.path.basename(model_id) if os.path.isabs(model_id) else model_id,
        )

    def build(self, generator=False, on_end=lambda: ()):
        engine_dir = os.path.join(self.model_dir, "engine")
        onnx_dir = os.path.join(self.model_dir, "onnx")
        os.makedirs(engine_dir, exist_ok=True)
        os.makedirs(onnx_dir, exist_ok=True)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            if generator:
                yield json.dumps(
                    {
                        "message": f"Building {model_name}...",
                        "progress": i / len(self.models.keys()),
                    }
                )
            engine = build_engine(
                model_name,
                engine_dir,
                onnx_dir,
                model_data,
                self.opt_image_height,
                self.opt_image_width,
                self.onnx_opset,
                self.force_engine_build,
                self.force_onnx_export,
                self.force_onnx_optimize,
                self.onnx_minimal_optimization,
                self.build_static_batch,
                self.build_dynamic_shape,
                self.build_preview_features,
            )
            del model_data
            del engine
        meta = {
            "models": {
                "unet": self.models["unet"].model_id,
                "vae": self.models["vae"].model_id,
            },
            "denoising_prec": "fp16" if self.fp16 else "fp32",
            "opt_image_height": self.opt_image_height,
            "opt_image_width": self.opt_image_width,
            "onnx_opset": self.onnx_opset,
            "build_static_batch": self.build_static_batch,
            "build_dynamic_shape": self.build_dynamic_shape,
        }
        txt = json.dumps(meta)
        with open(
            os.path.join(self.model_dir, "model_index.json"), mode="w", encoding="utf-8"
        ) as f:
            f.write(txt)
        on_end()
