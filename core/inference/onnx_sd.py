import importlib
import inspect
import logging
import os
import re
import shutil
import warnings
from dataclasses import fields
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import set_module_tensor_to_device
from diffusers import LMSDiscreteScheduler, SchedulerMixin
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.autoencoder_kl import AutoencoderKL, AutoencoderKLOutput
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.vae import DecoderOutput
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
)
from diffusers.utils import PIL_INTERPOLATION
from numpy.random import MT19937, RandomState, SeedSequence
from packaging import version
from PIL import Image
from torch.onnx import export
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import is_safetensors_available

from api import websocket_manager
from api.websockets import Data
from core.files import get_full_model_path
from core.inference.base_model import InferenceModel
from core.inference.functions import (
    is_onnx_available,
    is_onnxconverter_available,
    is_onnxscript_available,
    is_onnxsim_available,
)
from core.inference_callbacks import (
    img2img_callback,
    inpaint_callback,
    txt2img_callback,
)
from core.types import (
    Img2ImgQueueEntry,
    InpaintQueueEntry,
    Job,
    QuantizationDict,
    Txt2ImgQueueEntry,
)
from core.utils import convert_images_to_base64_grid, convert_to_image

logger = logging.getLogger(__name__)


class UNet2DConditionWrapper(UNet2DConditionModel):
    "Internal class"

    # I love being retarded
    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        _=None,
        __=None,
        ___=None,
        ____=None,
        _____=None,
        ______=None,
        _______: bool = True,
    ) -> Tuple:
        sample = sample.to(dtype=torch.float16)
        timestep = timestep.to(dtype=torch.long)  # type: ignore
        encoder_hidden_states = encoder_hidden_states.to(dtype=torch.float16)

        sample = UNet2DConditionModel.forward(
            self, sample, timestep, encoder_hidden_states, return_dict=True
        ).sample  # type: ignore
        return (sample.to(dtype=torch.float16),)


class CLIPTextModelWrapper(CLIPTextModel):
    "Internal class"

    def forward(self, input_ids) -> Tuple:
        outputs: BaseModelOutputWithPooling = CLIPTextModel.forward(
            self, input_ids=input_ids, return_dict=True
        )  # type: ignore
        return (
            outputs.last_hidden_state.to(dtype=torch.float32),
            outputs.pooler_output.to(dtype=torch.float32),
        )


class AutoencoderKLWrapper(AutoencoderKL):
    "Internal class"

    def encode(self, x) -> Tuple:  # pylint: disable=arguments-differ
        x = x.to(dtype=torch.float16)
        outputs: AutoencoderKLOutput = AutoencoderKL.encode(self, x, True)
        return (outputs.latent_dist.sample().to(dtype=torch.float32),)

    def decode(self, z) -> Tuple:  # pylint: disable=arguments-differ
        z = z.to(dtype=torch.float16)
        outputs: DecoderOutput = AutoencoderKL.decode(self, z, True)  # type: ignore
        return (outputs.sample.to(dtype=torch.float32),)


class OnnxStableDiffusion(InferenceModel):
    """
    Inference model capable of inpainting, img2img and txt2img.
    """

    def __init__(
        self,
        model_id: str,
        use_fp32: bool = False,
        autoload: bool = True,
    ) -> None:
        if is_onnx_available():
            import onnxruntime as ort

            super().__init__(model_id)
            self.use_fp32 = use_fp32
            self.vae_encoder: ort.InferenceSession
            self.vae_decoder: ort.InferenceSession
            self.unet: ort.InferenceSession
            self.text_encoder: ort.InferenceSession
            self.tokenizer: CLIPTokenizerFast
            self.scheduler: LMSDiscreteScheduler

            if autoload:
                self.load()
        else:
            raise ValueError("ONNX is not available")

    def load(self):
        if is_onnx_available():
            import onnxruntime as ort

            def _load(
                file: Path, providers: Optional[List[str]] = None
            ) -> Union[
                ort.InferenceSession,
                CLIPTokenizerFast,
                SchedulerMixin,
                Dict[str, List[str]],
            ]:
                if providers is None:
                    providers = ["CUDAExecutionProvider"]

                if file.stem == "providers":
                    with open(file, encoding="utf-8") as f:
                        # example providers.txt file:
                        #
                        # vae_encoder: CPUExecutionProvider
                        # vae_decoder: CPUExecutionProvider
                        # unet: CUDAExecutionProvider cudnn_conv_use_max_workspace enable_cuda_graph cudnn_conv1d_pad_to_nc1d
                        # text_encoder: CPUExecutionProvider
                        #

                        d = map(
                            lambda x: x.split(": "),
                            filter(lambda x: x != "", f.readlines()),
                        )
                        r = {}
                        for file_name, file_providers in d:
                            # There probably is a better way to do this, but honestly fuck it
                            r[file_name] = [
                                tuple(
                                    map(
                                        lambda x: {x: "1"}
                                        if "Provider" not in x
                                        else x,
                                        filter(
                                            lambda x: x != "", file_providers.split(" ")
                                        ),
                                    )
                                )
                            ]
                        for file_name, file_providers in r.items():
                            # So no "no fallback" warnings in console (it's ugly)
                            if "CPUExecutionProvider" not in file_providers:
                                file_providers.append("CPUExecutionProvider")
                                r[file_name] = file_providers
                        return r
                else:
                    if file.is_dir():
                        match file.stem:
                            case "tokenizer":
                                return CLIPTokenizerFast.from_pretrained(file)
                            case "scheduler":
                                # TODO: during conversion save which scheduler was used.
                                scheduler_reg = r"_class_name\": \"(.*)\","
                                with open(
                                    file / "scheduler_config.json",
                                    "r",
                                    encoding="utf-8",
                                ) as f:
                                    matches = re.search(
                                        scheduler_reg, "\n".join(f.readlines())
                                    )
                                    module = importlib.import_module("diffusers")
                                    assert (
                                        matches is not None
                                    ), "Scheduler not found in the scheduler_config.json"
                                    scheduler = getattr(module, matches.group(1))
                                    f = getattr(scheduler, "from_pretrained")
                                    return f(pretrained_model_name_or_path=str(file))
                            case _:
                                raise ValueError("Bad argument 'file' provided.")
                    else:
                        sess_options = ort.SessionOptions()

                        # TODO: benchmark me
                        sess_options.graph_optimization_level = (
                            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        )
                        if isinstance(providers[0], tuple):
                            provname = providers[0][0]
                        else:
                            provname = providers[0]
                        if provname == "DmlExecutionProvider":
                            sess_options.enable_mem_pattern = False
                            sess_options.execution_mode = (
                                ort.ExecutionMode.ORT_SEQUENTIAL
                            )
                        else:
                            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

                        return ort.InferenceSession(
                            str(file), providers=providers, sess_options=sess_options
                        )

            folder = Path("data/onnx").joinpath(self.model_id)

            if (folder / "providers.txt").exists():
                providers = _load(folder / "providers.txt")
            else:
                providers = {
                    "vae_encoder": ["CPUExecutionProvider"],
                    "vae_decoder": ["CPUExecutionProvider"],
                    "unet": ["CUDAExecutionProvider"],
                    "text_encoder": ["CPUExecutionProvider"],
                }

            for module in ["vae_encoder", "vae_decoder", "unet", "text_encoder"]:
                s = time()
                setattr(self, module, _load(folder / (module + ".onnx"), providers=providers[module]))  # type: ignore
                logger.info(f"Loaded {module} in {(time() - s):.2f}s.")
            for module in ["tokenizer", "scheduler"]:
                s = time()
                setattr(self, module, _load(folder / module))
                logger.info(f"Loaded {module} in {(time() - s):.2f}s.")

            self.memory_cleanup()

    def _setup(self, opset_version: int = 17):
        if is_onnxscript_available():
            # Until implemented in pytorch
            # https://github.com/pytorch/pytorch/issues/97262#issuecomment-1487141914
            # thank you @justinchuby

            import onnxscript  # pylint: disable=import-error,unreachable

            # make dynamic?
            from onnxscript.onnx_opset import opset17 as op  # pylint: disable=E0401

            custom_opset = onnxscript.values.Opset(
                domain="torch.onnx", version=opset_version
            )

            @onnxscript.script(custom_opset)
            def ScaledDotProductAttention(
                query,
                key,
                value,
                dropout_p,
            ):
                # Swap the last two axes of key
                key_shape = op.Shape(key)
                key_last_dim = key_shape[-1:]
                key_second_last_dim = key_shape[-2:-1]
                key_first_dims = key_shape[:-2]
                # Contract the dimensions that are not the last two so we can transpose
                # with a static permutation.
                key_squeezed_shape = op.Concat(
                    op.Constant(value_ints=[-1]),
                    key_second_last_dim,
                    key_last_dim,
                    axis=0,
                )
                key_squeezed = op.Reshape(key, key_squeezed_shape)
                key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
                key_transposed_shape = op.Concat(
                    key_first_dims, key_last_dim, key_second_last_dim, axis=0
                )
                key_transposed = op.Reshape(
                    key_squeezed_transposed, key_transposed_shape
                )

                embedding_size = op.CastLike(op.Shape(query)[-1], query)
                scale = op.Div(1.0, op.Sqrt(embedding_size))

                # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
                # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
                query_scaled = op.Mul(query, op.Sqrt(scale))
                key_transposed_scaled = op.Mul(key_transposed, op.Sqrt(scale))
                attn_weight = op.Softmax(
                    op.MatMul(query_scaled, key_transposed_scaled),
                    axis=-1,
                )
                attn_weight, _ = op.Dropout(attn_weight, dropout_p)
                return op.MatMul(attn_weight, value)

            def custom_scaled_dot_product_attention(
                g, query, key, value, attn_mask, dropout, is_causal, scale=None
            ):
                return g.onnxscript_op(
                    ScaledDotProductAttention, query, key, value, dropout
                ).setType(value.type())

            torch.onnx.register_custom_op_symbolic(
                symbolic_name="aten::scaled_dot_product_attention",
                symbolic_fn=custom_scaled_dot_product_attention,
                opset_version=opset_version,
            )

    def _run_model(self, model, **kwargs):
        inputs = {k: np.array(v) for k, v in kwargs.items()}

        # This was originally any(map(lambda x: x == "CUDAExecutionProvider", model.get_providers()))
        # I think I classify as a retard
        if (
            "CPUExecutionProvider" not in model.get_providers()
            and len(model.get_providers()) != 1
        ):
            iob = model.io_binding()
            for k, v in inputs.items():
                iob.bind_cpu_input(k, v)
            for k in model.get_outputs():
                iob.bind_output(k.name)
            model.run_with_iobinding(iob)
            return iob.copy_outputs_to_cpu()
        else:
            return model.run(None, inputs)

    @torch.no_grad()
    def convert_pytorch_to_onnx(
        self,
        model_id: str,
        target: Optional[QuantizationDict] = None,
        device: Union[torch.device, str] = "cuda",
        simplify_unet: bool = False,
    ):
        """
        Converts a pytorch model into an onnx model and tries to quantize it. Depending on model size, can take up to 6gigabytes of vram

        Keyword arguments:

        model  -- a repository, or a huggingface model name inside data/models/

        target -- default: {"vae_decoder": None, "vae_encoder": None, "unet": None, "text_encoder": None}. If any of them set to true, they will be quantized using uint8 instead of int8, and will be ran on cpu during inference. If set to false, quantization will be disabled.

        device -- default: "cuda." The device on which torch will run. If you have a gpu, you should use "cuda" or "cuda:0" or "cuda:1" etc. If you have a cpu, you should use "cpu". If on windows, and on amd, use torch_directml.device()

        simplify_unet -- default: False. Whether the UNet should be simplified (this uses upwards of 20gb of RAM)
        """

        if target is None:
            target = QuantizationDict(
                vae_encoder=None, vae_decoder=None, unet=None, text_encoder=None
            )

        def onnx_export(
            model,
            model_args: Tuple,
            output_path: Path,
            ordered_input_names: List[str],
            output_names: List[str],
            dynamic_axes: Dict,
            signed: bool,
            opset: int,
            simplify: Optional[bool] = True,
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rt = time()
            logger.info("Starting export on model %s", output_path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                export(
                    model,
                    model_args,
                    f=output_path.as_posix(),
                    input_names=ordered_input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                    opset_version=opset,
                )
                logger.info(
                    "Finished initial export of model %s in %.2fs",
                    output_path,
                    time() - rt,
                )
            # quantize
            quantize_success = False if signed is not None else True
            try:
                if signed is not None:
                    from onnxruntime.quantization import QuantType, quantize_dynamic

                    t = time()
                    logger.info(
                        "Starting quantization process of model %s", output_path
                    )
                    in_path = output_path
                    output_path = output_path.with_stem(output_path.stem + ".quant")
                    old_size = os.stat(str(in_path)).st_size
                    quantize_dynamic(
                        in_path,
                        output_path,
                        weight_type=QuantType.QUInt8 if signed else QuantType.QInt8,
                        per_channel=True,
                        reduce_range=True,
                    )
                    new_size = os.stat(str(output_path)).st_size
                    os.remove(in_path)
                    os.rename(output_path, in_path)

                    output_path = in_path
                    quantize_success = True
                    logger.info(
                        "Quantization successful on model %s in %.2fs. %.2fgb - %.2fgb -- %.2f%% change)",
                        output_path,
                        time() - t,
                        old_size / (1024**3),
                        new_size / (1024**3),
                        ((old_size / new_size) - 1) * 100,
                    )

                    self.memory_cleanup()
            except ValueError:
                output_path = in_path  # type: ignore
                logger.warning("Could not quantize model, skipping.")

            self.memory_cleanup()

            if is_onnxconverter_available():
                if not quantize_success and not signed and not self.use_fp32:
                    logger.info(f"Starting FP16 conversion on {str(output_path)}")
                    t = time()
                    import onnx
                    import onnxruntime as ort  # pylint: disable=import-error
                    from onnxconverter_common import float16  # pylint: disable=E0401

                    model = onnx.load(str(output_path))  # pylint: disable=no-member
                    model = float16.convert_float_to_float16(model, keep_io_types=True)
                    onnx.save(model, str(output_path))  # pylint: disable=no-member
                    del model
                    logger.info(
                        f"Conversion successful on model {str(output_path)} in {(time() - t):.2f}s."
                    )
            elif not self.use_fp32 and not quantize_success:
                logger.warning(
                    "Onnxconverter-common is not available, skipping float16 conversion process. Model will run in FP32."
                )

            self.memory_cleanup()

            if simplify:
                if is_onnxsim_available():
                    logger.info("Starting simplification process on %s", output_path)
                    try:
                        import onnx  # pylint: disable=import-self
                        import onnxsim as onx  # pylint: disable=import-error

                        t = time()
                        model = onnx.load(str(output_path))  # pylint: disable=no-member
                        model_opt, check = onx.simplify(model)  # type: ignore
                        if not check:
                            logger.warning(
                                f"Could not validate simplified model at path {output_path}"
                            )
                            del model, model_opt, check
                            raise ValueError("Could not validate simplified model")
                        del model
                        old_size = os.stat(str(output_path)).st_size
                        os.remove(str(output_path))
                        onnx.save(  # pylint: disable=no-member
                            model_opt, str(output_path)
                        )  # pylint: disable=no-member
                        new_size = os.stat(str(output_path)).st_size
                        del model_opt

                        logger.info(
                            "Simplification successful on model %s in %.2fs. %.2fgb - %.2fgb -- %.2f%% change)",
                            output_path,
                            time() - t,
                            old_size / (1024**3),
                            new_size / (1024**3),
                            ((old_size / new_size) - 1) * 100,
                        )
                        self.memory_cleanup()
                    except ValueError:
                        logger.warning("Could not simplify model, skipping.")
                else:
                    logger.warning(
                        "Onnx-simplifier is not available, skipping simplification process."
                    )

            logger.info(
                "Finished exporting %s. Total time: %.2fs", output_path, time() - rt
            )
            self.memory_cleanup()

        T = TypeVar("T")

        def load(
            cls: Type[T],
            root: Path,
            checkpoint_name="diffusion_pytorch_model.bin",
            dtype=None,
        ) -> T:
            with init_empty_weights():
                model = cls.from_config(  # type: ignore
                    config_path=root / "config.json", pretrained_model_name_or_path=root
                )

            if checkpoint_name.endswith(".safetensors"):
                if is_safetensors_available():
                    from safetensors.torch import load_file

                    state_dict = load_file(
                        str(root.joinpath(checkpoint_name)), device="cpu"
                    )
                    for name, param in state_dict.items():
                        set_module_tensor_to_device(
                            model, name, device, value=param, dtype=dtype
                        )
                else:
                    logger.error("Cannot load safetensors without safetensors package.")
            else:
                load_checkpoint_and_dispatch(
                    model, str(root / checkpoint_name), device_map="auto"
                )
                model = model.to(dtype=dtype, device=device)

                model.eval()
            return model

        def convert_text_encoder(
            main_folder: Path,
            output_folder: Path,
            tokenizer: CLIPTokenizerFast,
            opset: int,
        ) -> Tuple[int, int]:
            text_encoder = CLIPTextModelWrapper.from_pretrained(
                main_folder / "text_encoder"
            )
            assert isinstance(text_encoder, CLIPTextModelWrapper)
            text_encoder.to(dtype=torch.float16, device=device)
            text_encoder.eval()

            num_tokens = text_encoder.config.max_position_embeddings
            text_hidden_size = text_encoder.config.hidden_size
            max_length = tokenizer.model_max_length

            text_input = tokenizer(
                "You miss 100% of the shots you don't take, but you also don't shoot them all, even when you do.",
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device=device, dtype=torch.int32)
            onnx_export(
                text_encoder,
                model_args=(text_input,),
                output_path=output_folder / "text_encoder.onnx",
                ordered_input_names=["input_ids"],
                output_names=["last_hidden_state", "pooler_output"],
                dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
                opset=opset,
                signed=target.text_encoder,  # type: ignore
            )

            del text_encoder
            self.memory_cleanup()
            return num_tokens, text_hidden_size

        def convert_unet(
            main_folder: Path,
            output_folder: Path,
            num_tokens: int,
            text_hidden_size: int,
            opset: int,
        ) -> int:
            unet = load(UNet2DConditionWrapper, main_folder / "unet", dtype=torch.float16)  # type: ignore

            if version.parse(torch.__version__) > version.parse("2.0.0"):
                from diffusers.models.attention_processor import AttnProcessor2_0

                logger.info("Compiling SDPA into model")
                unet.set_attn_processor(AttnProcessor2_0())  # type: ignore
            else:
                logger.info("Compiling cross-attention into model")
                unet.set_attn_processor(AttnProcessor())

            if isinstance(unet.config.attention_head_dim, int):  # type: ignore pylint: disable=no-member
                slice_size = unet.config.attention_head_dim // 2  # type: ignore pylint: disable=no-member
            else:
                slice_size = min(unet.config.attention_head_dim)  # type: ignore pylint: disable=no-member
            unet.set_attention_slice(slice_size)

            unet_model_size = 0
            for param in unet.parameters():
                unet_model_size += param.nelement() * param.element_size()
            for buffer in unet.buffers():
                unet_model_size += buffer.nelement() * buffer.element_size()
            unet_model_size_mb = unet_model_size / 1024**2
            needs_collate = unet_model_size_mb > 2000

            in_channels = unet.config["in_channels"]
            sample_size = unet.config["sample_size"]

            unet_out_path = output_folder
            if needs_collate:
                unet_out_path = output_folder / "unet_data"
                unet_out_path.mkdir(parents=True, exist_ok=True)
            onnx_export(
                unet,
                model_args=(
                    torch.randn(2, in_channels, sample_size, sample_size + 1).to(
                        device=device, dtype=torch.float32
                    ),
                    torch.randn(2).to(device=device, dtype=torch.float32),
                    torch.randn(2, (num_tokens * 3) - 2, text_hidden_size).to(
                        device=device, dtype=torch.float32
                    ),
                ),
                output_path=unet_out_path / "unet.onnx",
                ordered_input_names=[
                    "sample",
                    "timestep",
                    "encoder_hidden_states",
                    "return_dict",
                ],
                output_names=["out_sample"],
                dynamic_axes={
                    "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                    "timestep": {0: "batch"},
                    "encoder_hidden_states": {0: "batch", 1: "sequence"},
                },
                opset=opset,
                simplify=simplify_unet,
                signed=target.unet,  # type: ignore
            )
            del unet
            self.memory_cleanup()
            if needs_collate:
                unet = onnx.load(  # type: ignore pylint: disable=undefined-variable
                    str((unet_out_path / "unet.onnx").absolute().as_posix())
                )
                onnx.save_model(  # type: ignore pylint: disable=undefined-variable
                    unet,
                    str((output_folder / "unet.onnx").absolute().as_posix()),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="unet.pb",
                    convert_attribute=False,
                )

                del unet
                self.memory_cleanup()
                shutil.rmtree(unet_out_path)
                unet_out_path = output_folder
            return sample_size

        def convert_vae(
            main_folder: Path, output_folder: Path, unet_sample_size: int, opset: int
        ):
            vae = load(AutoencoderKLWrapper, main_folder / "vae", dtype=torch.float16)  # type: ignore
            vae_in_channels = vae.config["in_channels"]
            vae_sample_size = vae.config["sample_size"]
            vae_latent_channels = vae.config["latent_channels"]

            vae.forward = lambda sample: vae.encode(sample)[0]  # type: ignore
            onnx_export(
                vae,
                model_args=(
                    torch.randn(
                        1, vae_in_channels, vae_sample_size, vae_sample_size
                    ).to(device=device, dtype=torch.float32),
                ),
                output_path=output_folder / "vae_encoder.onnx",
                ordered_input_names=["sample"],
                output_names=["latent_sample"],
                dynamic_axes={
                    "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"}
                },
                opset=opset,
                signed=target.vae_encoder,  # type: ignore
            )

            vae.forward = vae.decode  # type: ignore
            onnx_export(
                vae,
                model_args=(
                    torch.randn(
                        1, vae_latent_channels, unet_sample_size, unet_sample_size
                    ).to(device=device, dtype=torch.float32),
                ),
                output_path=output_folder / "vae_decoder.onnx",
                ordered_input_names=["latent_sample"],
                output_names=["sample"],
                dynamic_axes={
                    "latent_sample": {
                        0: "batch",
                        1: "channels",
                        2: "height",
                        3: "width",
                    }
                },
                opset=opset,
                signed=target.vae_decoder,  # type: ignore
            )
            del vae
            self.memory_cleanup()

        opset = 17
        main_folder = get_full_model_path(model_id)
        model_id_fixed = model_id.replace("/", "--")
        output_folder = Path(f"data/onnx/{model_id_fixed}")

        if is_onnx_available():
            # register aten::scaled_dot_product_attention
            self._setup(opset)

            tokenizer = CLIPTokenizerFast.from_pretrained(main_folder / "tokenizer")

            num_tokens, text_hidden_size = convert_text_encoder(
                main_folder, output_folder, tokenizer, opset
            )
            unet_sample_size = convert_unet(
                main_folder, output_folder, num_tokens, text_hidden_size, opset
            )
            convert_vae(main_folder, output_folder, unet_sample_size, opset)
            shutil.copytree(main_folder / "tokenizer", output_folder / "tokenizer")
            shutil.copytree(main_folder / "scheduler", output_folder / "scheduler")

            with open(output_folder / "providers.txt", mode="x", encoding="utf-8") as f:
                for field in fields(target):
                    fn, prov = field.name, getattr(target, field.name)
                    if fn == "unet":
                        t = (
                            "CPUExecutionProvider"
                            if prov
                            else "CUDAExecutionProvider cudnn_conv_use_max_workspace enable_cuda_graph cudnn_conv1d_pad_to_nc1d"
                        )
                    else:
                        t = "CPUExecutionProvider" if prov else "CUDAExecutionProvider"
                    f.write(f"{fn}: {t} \n")

    def _encode_prompt(
        self,
        prompt: str,
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[str],
    ):
        t = time()
        logger.debug("encode prompt")
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        ).input_ids

        prompt_embeds = self._run_model(
            self.text_encoder, input_ids=text_input_ids.astype(np.int32)
        )[0]
        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            negative_prompt_embeds = self._run_model(
                self.text_encoder, input_ids=uncond_input.input_ids.astype(np.int32)
            )[0]
            negative_prompt_embeds = np.repeat(
                negative_prompt_embeds, num_images_per_prompt, axis=0
            )

            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])
        logger.debug("encode prompt end (%.2fs)", time() - t)
        return prompt_embeds

    def _get_timestep_dtype(self):
        timestep_dtype = next(
            (
                input.type
                for input in self.unet.get_inputs()
                if input.name == "timestep"
            ),
            "tensor(float)",
        )
        return ORT_TO_NP_TYPE[timestep_dtype]

    def _timestep(
        self,
        latents,
        timestep_dtype,
        do_classifier_free_guidance,
        guidance_scale,
        prompt_embeds,
        extra_step_kwargs,
        callback=None,
        timesteps=None,
        kw=None,
        class_labels=None,
    ):
        def _init_latent_model(latents, do_classifier_free_guidance, t):
            latent_model_input_numpy = (
                np.concatenate([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(
                torch.from_numpy(latent_model_input_numpy), t  # type: ignore
            )
            latent_model_input = latent_model_input.cpu().numpy()
            return latent_model_input

        logger.debug("timestep start")
        rt = time()
        for i, t in enumerate(
            tqdm(self.scheduler.timesteps if timesteps is None else timesteps)
        ):
            if kw is not None:
                latent_model_input = kw(latents, do_classifier_free_guidance, t)
            else:
                latent_model_input = _init_latent_model(
                    latents, do_classifier_free_guidance, t
                )

            timestep = np.array([t], dtype=timestep_dtype)
            if class_labels is None:  # rookie mistake
                noise_pred = self._run_model(
                    self.unet,
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                )
            else:
                noise_pred = self._run_model(
                    self.unet,
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=class_labels,
                )
            noise_pred = noise_pred[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred),  # type: ignore
                t,  # type: ignore
                torch.from_numpy(latents),  # type: ignore
                **extra_step_kwargs,
            )
            latents = scheduler_output.prev_sample  # type: ignore

            if callback is not None:
                callback(i, t, latents)
            latents = latents.numpy()
        logger.debug("timestep end (%.2fs)", time() - rt)
        return latents

    def _extra_args(self):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = 0.0
        return extra_step_kwargs

    def _decode_latents(self, latents, latent_value):
        latents = 1 / latent_value * latents
        t = time()
        logger.debug("vae_decoder start")
        image = np.concatenate(
            [
                self._run_model(self.vae_decoder, latent_sample=latents[i : i + 1])[0]
                for i in range(latents.shape[0])
            ]
        )
        logger.debug("vae_decoder end (%.2fs)", time() - t)
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        return image

    @torch.no_grad()
    # broken
    # TODO: fix later
    def _inpaint(
        self,
        prompt: str,
        image: Image.Image,
        mask_image: Image.Image,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,  # type: ignore
        num_images_per_prompt: int = 1,
        callback=None,
        seed: int = -1,
        generator: RandomState = None,  # type: ignore pylint: disable=no-member
        latents: np.ndarray = None,  # type: ignore
    ) -> StableDiffusionPipelineOutput:
        if generator is None:
            generator = RandomState(MT19937(SeedSequence(seed)))
        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps)
        prompt_embeds = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        latents_shape = (num_images_per_prompt, 4, height // 8, width // 8)
        latents_dtype = prompt_embeds.dtype
        if latents is None:
            latents = generator.randn(*latents_shape).astype(latents_dtype)

        # preprocess
        latents_shape = latents_shape[-2:]
        mask = mask_image
        image = np.array(  # type: ignore
            image.convert("RGB").resize((latents_shape[1] * 8, latents_shape[0] * 8))
        )
        image = image[None].transpose(0, 3, 1, 2)  # type: ignore
        image = image.astype(np.float32) / 127.5 - 1.0  # type: ignore

        image_mask = np.array(
            mask.convert("L").resize((latents_shape[1] * 8, latents_shape[0] * 8))
        )
        masked_image = image * (image_mask < 127.5)  # type: ignore

        mask = mask.resize(
            (latents_shape[1], latents_shape[0]), PIL_INTERPOLATION["nearest"]
        )
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None, None]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        mask = mask.astype(latents.dtype)
        masked_image = masked_image.astype(latents.dtype)

        masked_image_latents = self._run_model(self.vae_encoder, sample=masked_image)[0]
        masked_image_latents = 0.18215 * masked_image_latents

        mask = mask.repeat(num_images_per_prompt, 0)
        masked_image_latents = masked_image_latents.repeat(num_images_per_prompt, 0)

        mask = np.concatenate([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            np.concatenate([masked_image_latents] * 2)
            if do_classifier_free_guidance
            else masked_image_latents
        )

        latents = latents * np.float64(self.scheduler.init_noise_sigma)
        extra_step_kwargs = self._extra_args()
        timestep_dtype = self._get_timestep_dtype()

        def _init_latent_model(latents, do_classifier_free_guidance, t):
            latent_model_input = (
                np.concatenate([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(
                torch.from_numpy(latent_model_input), t  # type: ignore
            )
            latent_model_input = latent_model_input.cpu().numpy()
            latent_model_input = np.concatenate(
                [latent_model_input, mask, masked_image_latents], axis=1
            )
            return latent_model_input

        latents = self._timestep(
            latents=latents,
            timestep_dtype=timestep_dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            extra_step_kwargs=extra_step_kwargs,
            callback=callback,
            kw=_init_latent_model,
        )

        image = self._decode_latents(latents, 0.18215)  # type: ignore
        image = self.numpy_to_pil(image)  # type: ignore
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)  # type: ignore

    @torch.no_grad()
    def _txt2img(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,  # type: ignore
        num_images_per_prompt: int = 1,
        callback=None,
        seed: int = -1,
        generator: RandomState = None,  # type: ignore pylint: disable=no-member
        latents: np.ndarray = None,  # type: ignore
    ) -> StableDiffusionPipelineOutput:
        if generator is None:
            generator = RandomState(MT19937(SeedSequence(seed)))
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        latents_dtype = prompt_embeds.dtype
        latents_shape = (num_images_per_prompt, 4, height // 8, width // 8)
        latents = generator.randn(*latents_shape).astype(latents_dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        extra_step_kwargs = self._extra_args()
        timestep_dtype = self._get_timestep_dtype()

        latents = self._timestep(
            latents=latents,
            timestep_dtype=timestep_dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            extra_step_kwargs=extra_step_kwargs,
            callback=callback,
        )

        image = self._decode_latents(latents, 0.18215)
        image = self.numpy_to_pil(image)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)  # type: ignore

    @torch.no_grad()
    def _img2img(
        self,
        prompt: str,
        image: Image.Image,
        height: int = 512,
        width: int = 512,
        strength: float = 0.8,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        callback=None,
        seed: int = -1,
    ) -> StableDiffusionPipelineOutput:
        do_classifier_free_guidance = guidance_scale > 1.0

        width, height = (
            x - x % 64 for x in (width, height)
        )  # resize to integer multiple of 64

        image = [image]  # type: ignore
        image = [np.array(i.resize((width, height)))[None, :] for i in image]  # type: ignore
        image = np.concatenate(image, axis=0)  # type: ignore
        image = np.array(image).astype(np.float32) / 255.0  # type: ignore
        image = image.transpose(0, 3, 1, 2)  # type: ignore
        image = 2.0 * image - 1.0  # type: ignore

        self.scheduler.set_timesteps(num_inference_steps)
        generator = RandomState(MT19937(SeedSequence(seed)))

        prompt_embeds = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        latents_dtype = prompt_embeds.dtype

        init_latents = self._run_model(self.vae_encoder, sample=image)[0]
        init_latents = 0.18215 * init_latents
        init_latents = np.concatenate([init_latents] * num_images_per_prompt, axis=0)

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        timesteps = self.scheduler.timesteps.numpy()[-init_timestep]
        timesteps = np.array([timesteps] * num_images_per_prompt)

        noise = generator.randn(*init_latents.shape).astype(latents_dtype)  # type: ignore
        init_latents = self.scheduler.add_noise(
            torch.from_numpy(init_latents),  # type: ignore
            torch.from_numpy(noise),  # type: ignore
            torch.from_numpy(timesteps),  # type: ignore
        ).numpy()

        t_start = max(0, num_inference_steps - init_timestep)
        timesteps = self.scheduler.timesteps[t_start:].numpy()

        extra_step_kwargs = self._extra_args()
        timestep_dtype = self._get_timestep_dtype()

        latents = self._timestep(
            latents=init_latents,
            timestep_dtype=timestep_dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            extra_step_kwargs=extra_step_kwargs,
            callback=callback,
            timesteps=timesteps,
        )

        image = self._decode_latents(latents, 0.18215)  # type: ignore
        image = self.numpy_to_pil(image)  # type: ignore
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)  # type: ignore

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [
                Image.fromarray(image.squeeze(), mode="L") for image in images
            ]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def unload(self):
        del (
            self.vae_encoder,
            self.vae_decoder,
            self.unet,
            self.text_encoder,
            self.tokenizer,
            self.scheduler,
        )
        self.memory_cleanup()

    def generate(self, job: Job) -> List[Image.Image]:
        total_images = []
        if isinstance(job, Txt2ImgQueueEntry):
            for _ in range(job.data.batch_count):
                total_images.extend(
                    self._txt2img(
                        job.data.prompt,
                        job.data.height,
                        job.data.width,
                        job.data.steps,
                        job.data.guidance_scale,
                        job.data.negative_prompt,
                        job.data.batch_size,
                        txt2img_callback,
                        job.data.seed,
                    ).images
                )
            websocket_manager.broadcast_sync(
                data=Data(
                    data_type="txt2img",
                    data={
                        "progress": 0,
                        "current_step": 0,
                        "total_steps": 0,
                        "image": convert_images_to_base64_grid(
                            total_images, quality=90, image_format="webp"
                        ),
                    },
                )
            )
        elif isinstance(job, Img2ImgQueueEntry):
            for _ in range(job.data.batch_count):
                total_images.extend(
                    self._img2img(
                        job.data.prompt,
                        convert_to_image(job.data.image),
                        job.data.height,
                        job.data.width,
                        job.data.strength,
                        job.data.steps,
                        job.data.guidance_scale,
                        job.data.negative_prompt,
                        job.data.batch_size,
                        img2img_callback,
                        job.data.seed,
                    ).images
                )
            websocket_manager.broadcast_sync(
                data=Data(
                    data_type="img2img",
                    data={
                        "progress": 0,
                        "current_step": 0,
                        "total_steps": 0,
                        "image": convert_images_to_base64_grid(
                            total_images, quality=90, image_format="webp"
                        ),
                    },
                )
            )
        elif isinstance(job, InpaintQueueEntry):
            for _ in range(job.data.batch_count):
                total_images.extend(
                    self._inpaint(
                        job.data.prompt,
                        convert_to_image(job.data.image),
                        convert_to_image(job.data.mask_image),
                        job.data.height,
                        job.data.width,
                        job.data.steps,
                        job.data.guidance_scale,
                        job.data.negative_prompt,
                        job.data.batch_size,
                        inpaint_callback,
                        job.data.seed,
                    ).images
                )
            websocket_manager.broadcast_sync(
                data=Data(
                    data_type="inpainting",
                    data={
                        "progress": 0,
                        "current_step": 0,
                        "total_steps": 0,
                        "image": convert_images_to_base64_grid(
                            total_images, quality=90, image_format="webp"
                        ),
                    },
                )
            )
        else:
            raise ValueError("Invalid job type for this pipeline")

        return total_images
