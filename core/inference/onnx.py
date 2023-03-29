from pathlib import Path

from core.inference.base_model import InferenceModel

import logging

import os
import shutil

from tqdm.auto import tqdm

from PIL import Image
from typing import Any, List
from core.files import get_full_model_path
from transformers import CLIPTokenizer
from diffusers import LMSDiscreteScheduler
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion import OnnxStableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers import OnnxRuntimeModel
from core.types import (
    Job,
    Txt2ImgQueueEntry,
)
import torch
from torch.onnx import export
import onnx
from onnxruntime.quantization import quantize_dynamic

import inspect
import numpy as np

logger = logging.getLogger(__name__)

def is_onnx_available():
    try:
        import onnx
        import onnxruntime
        return True
    except ImportError:
        return False

def is_onnxscript_available():
    try:
        import onnxscript
        return True
    except ImportError:
        return False

class OnnxStableDiffusion(InferenceModel):
    def __init__(
        self,
        model_id: str,
        use_f32: bool = False,
        device: str = "cuda",
        autoload: bool = True,
    ) -> None:
        if is_onnx_available():
            super().__init__(model_id)
            self.vae_encoder: OnnxRuntimeModel
            self.vae_decoder: OnnxRuntimeModel
            self.unet: OnnxRuntimeModel
            self.text_encoder: OnnxRuntimeModel
            self.tokenizer: CLIPTokenizer
            self.scheduler: Any
            self.feature_extractor: Any
            self.requires_safety_checker: bool
            self.safety_checker: Any
        else:
            raise ValueError("ONNX is not available")

    def load(self):
        provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        folder = get_full_model_path(self.model_id, model_folder="onnx", force=True)

        pipeline = OnnxStableDiffusionPipeline.from_pretrained(folder, provider=provider)

        for m in ["vae_encoder", "vae_decoder", "unet", "text_encoder", "tokenizer", "scheduler", "safety_checker", "feature_extractor"]:
            setattr(self, m, getattr(pipeline, m))

        self.requires_safety_checker = False
        self.memory_cleanup()

    # Implement later, since there's some bullshit error I don't understand..????
    def _setup(self, opset_version: int = 17):
        if is_onnxscript_available():
            import onnxscript # pylint: disable=import-error,unreachable

            # make dynamic?
            from onnxscript.onnx_opset import opset17 as op

            custom_opset = onnxscript.values.Opset(domain="torch.onnx", version=opset_version)


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
                    op.Constant(value_ints=[-1]), key_second_last_dim, key_last_dim, axis=0
                )
                key_squeezed = op.Reshape(key, key_squeezed_shape)
                key_squeezed_transposed = op.Transpose(key_squeezed, perm=[0, 2, 1])
                key_transposed_shape = op.Concat(key_first_dims, key_last_dim, key_second_last_dim, axis=0)
                key_transposed = op.Reshape(key_squeezed_transposed, key_transposed_shape)

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


            def custom_scaled_dot_product_attention(g, query, key, value, attn_mask, dropout, is_causal, scale=None):
                return g.onnxscript_op(ScaledDotProductAttention, query, key, value, dropout).setType(V.type())


            torch.onnx.register_custom_op_symbolic(
                symbolic_name="aten::scaled_dot_product_attention",
                symbolic_fn=custom_scaled_dot_product_attention,
                opset_version=opset_version,
            )

    @torch.no_grad()
    def convert_pytorch_to_onnx(self, model_id: str):
        def onnx_export(
            model,
            model_args: tuple,
            output_path: Path,
            ordered_input_names,
            output_names,
            dynamic_axes,
            opset
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
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
            # quantize
            try:
                quantize_dynamic(
                    output_path,
                    Path(output_path.as_posix() + ".quant"),
                    use_external_data_format=True,
                    nodes_to_exclude=["InsertedCast_onnx::Conv_250", "InsertedCast_onnx::Conv_737", "InsertedCast_sample", "InsertedCast_latent_sample"],
                    per_channel=True,
                    reduce_range=True,
                )
            except ValueError:
                logger.warning("Could not quantize model, skipping.")

        opset = 17
        # register aten::scaled_dot_product_attention
        self._setup(opset)

        dtype = torch.float16
        output_path = get_full_model_path(model_id, model_folder="onnx", force=True)
        device = "cuda"
        pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path=get_full_model_path(model_id), torch_dtype=dtype)
        pipeline.enable_sequential_cpu_offload() # type: ignore
        num_tokens = pipeline.text_encoder.config.max_position_embeddings # type: ignore
        text_hidden_size = pipeline.text_encoder.config.hidden_size # type: ignore
        text_input = pipeline.tokenizer( # type: ignore
            "Text is good",
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length, # type: ignore
            truncation=True,
            return_tensors="pt",
        )
        onnx_export(
            pipeline.text_encoder, # type: ignore
            # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
            model_args=(text_input.input_ids.to(device=device, dtype=torch.int32)),
            output_path=output_path / "text_encoder" / "model.onnx",
            ordered_input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
            },
            opset=opset,
        )
        del pipeline.text_encoder # type: ignore

        # UNET
        # Fix for PyTorch 2.0
        from diffusers.models.cross_attention import CrossAttnProcessor
        pipeline.unet.set_attn_processor(CrossAttnProcessor()) # type: ignore

        unet_in_channels = pipeline.unet.config.in_channels # type: ignore
        unet_sample_size = pipeline.unet.config.sample_size # type: ignore
        unet_path = output_path / "unet" / "model.onnx"
        onnx_export(
            pipeline.unet, # type: ignore
            model_args=(
                torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.randn(2).to(device=device, dtype=dtype),
                torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
                False,
            ),
            output_path=unet_path,
            ordered_input_names=["sample", "timestep", "encoder_hidden_states", "return_dict"],
            output_names=["out_sample"],  # has to be different from "sample" for correct tracing
            dynamic_axes={
                "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "timestep": {0: "batch"},
                "encoder_hidden_states": {0: "batch", 1: "sequence"},
            },
            opset=opset
        )
        unet_model_path = str(unet_path.absolute().as_posix())
        unet_dir = os.path.dirname(unet_model_path)
        unet = onnx.load(unet_model_path)
        # clean up existing tensor files
        shutil.rmtree(unet_dir)
        os.mkdir(unet_dir)
        # collate external tensor files into one
        onnx.save_model(
            unet,
            unet_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        del pipeline.unet # type: ignore

        # VAE ENCODER
        vae_encoder = pipeline.vae # type: ignore
        vae_in_channels = vae_encoder.config.in_channels
        vae_sample_size = vae_encoder.config.sample_size
        # need to get the raw tensor output (sample) from the encoder
        vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(sample, return_dict)[0].sample()
        onnx_export(
            vae_encoder,
            model_args=(
                torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(device=device, dtype=dtype),
                False,
            ),
            output_path=output_path / "vae_encoder" / "model.onnx",
            ordered_input_names=["sample", "return_dict"],
            output_names=["latent_sample"],
            dynamic_axes={
                "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=opset,
        )

        # VAE DECODER
        vae_decoder = pipeline.vae # type: ignore
        vae_latent_channels = vae_decoder.config.latent_channels
        # forward only through the decoder part
        vae_decoder.forward = vae_encoder.decode
        onnx_export(
            vae_decoder,
            model_args=(
                torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                False,
            ),
            output_path=output_path / "vae_decoder" / "model.onnx",
            ordered_input_names=["latent_sample", "return_dict"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=opset,
        )
        del pipeline.vae # type: ignore

    def _encode_prompt(self, prompt: str, num_images_per_prompt: int, do_classifier_free_guidance: bool, negative_prompt: str):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np"
        ).input_ids

        # Change me?
        prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]
        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="np")

            # Change me?
            negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            prompt_embeds = np.concatenate([negative_prompt, prompt_embeds])
        return prompt_embeds
    
    @torch.no_grad()
    def _text2img(
            self,
            prompt: str,
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 20,
            guidance_scale: float = 7.5,
            negative_prompt: str = None,
            num_images_per_prompt: int = 1,
            generator: np.random.RandomState | None = None,
            eta: float = 0.0,
            latents: np.ndarray | None = None,
    ):
        batch_size = 1
        if generator is None:
            generator = np.random
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, height // 8, width // 8)
        latents = generator.randn(*latents_shape).astype(latents_dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)
            noise_pred = noise_pred[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            scheduler_output = self.scheduler.step(
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()
        latents = 1 / 0.18215 * latents
        image = np.concatenate(
            [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
        )
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = image.transpose((0, 2, 3, 1))
        image = self.numpy_to_pil(image)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=False)

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
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
            self.feature_extractor,
            self.requires_safety_checker,
            self.safety_checker,
        )
        self.memory_cleanup()

    def text2img(self, job: Txt2ImgQueueEntry) -> List[Image.Image]:
        return self.generate(job)

    def generate(self, job: Job) -> List[Image.Image]:
        if isinstance(job, Txt2ImgQueueEntry):
            return self.text2img(job)
        return []
