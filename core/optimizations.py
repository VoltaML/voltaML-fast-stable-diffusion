import math
import logging
import types
from typing import Union, Optional, Dict, Tuple, Any

import torch
from torch.ao.nn.quantized import QFunctional
from tqdm import tqdm
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
)
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
import diffusers.models.embeddings as emb
from diffusers.utils import is_accelerate_available, is_xformers_available
from packaging import version

from core.config import config
from core.files import get_full_model_path

logger = logging.getLogger(__name__)

USE_DISK_OFFLOAD = False
quantize = False

gpu_module = None
_device = None

# torch.fx.wrap("len")


def optimize_model(
    pipe: Union[StableDiffusionPipeline, StableDiffusionUpscalePipeline],
    device,
    use_fp32: bool,
    is_for_aitemplate: bool = False,
) -> None:
    "Optimize the model for inference"
    global _device  # pylint: disable=global-statement

    pipe.to(device, torch_dtype=torch.float16 if not use_fp32 else torch.float32)
    _device = device

    logger.info("Optimizing model")

    # Attention slicing that should save VRAM (but is slower)
    slicing = config.api.attention_slicing
    if slicing != "disabled" and is_pytorch_pipe(pipe) and not is_for_aitemplate:
        if slicing == "auto":
            pipe.enable_attention_slicing()
            logger.info("Optimization: Enabled attention slicing")
        else:
            pipe.enable_attention_slicing(slicing)
            logger.info(f"Optimization: Enabled attention slicing ({slicing})")

    # Change the order of the channels to be more efficient for the GPU
    # DirectML only supports contiguous memory format
    if (
        config.api.channels_last
        and config.api.device != "directml"
        and not is_for_aitemplate
    ):
        pipe.unet.to(memory_format=torch.channels_last)  # type: ignore
        pipe.vae.to(memory_format=torch.channels_last)  # type: ignore
        logger.info("Optimization: Enabled channels_last memory format")

    # xFormers and SPDA
    if not is_for_aitemplate and not quantize:
        if is_xformers_available() and config.api.attention_processor == "xformers":
            if config.api.trace_model:
                logger.info("Optimization: Tracing model")
                pipe.unet = trace_model(pipe.unet)  # type: ignore
                logger.info("Optimization: Model successfully traced")
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Optimization: Enabled xFormers memory efficient attention")
        elif version.parse(torch.__version__) >= version.parse("2.0.0"):
            from diffusers.models.attention_processor import AttnProcessor2_0

            pipe.unet.set_attn_processor(AttnProcessor2_0())  # type: ignore
            logger.info("Optimization: Enabled SDPA, because xformers is not installed")
        else:
            # This should only be the case if pytorch_directml is to be used
            # This isn't a hot-spot either, so it's fine (imo) to put in safety nets.
            from diffusers.models.attention_processor import AttnProcessor

            pipe.unet.set_attn_processor(AttnProcessor())  # type: ignore
            logger.info(
                "Optimization: Pytorch STILL not newer than 2.0.0, using Cross-Attention"
            )

    offload = (
        config.api.offload
        if (is_pytorch_pipe(pipe) and not is_for_aitemplate)
        else None
    )
    if config.api.device != "directml":
        if offload == "model":
            # Offload to CPU

            pipe.vae.to("cpu")  # type: ignore
            pipe.unet.to("cpu")  # type: ignore
            pipe.unet.register_forward_pre_hook(send_to_gpu)  # type: ignore
            pipe.vae.register_forward_pre_hook(send_to_gpu)  # type: ignore
            setattr(pipe.vae, "main_device", True)  # type: ignore
            setattr(pipe.unet, "main_device", True)  # type: ignore
            logger.info("Optimization: Offloaded VAE & UNet to CPU.")

        elif offload == "module":
            # Enable sequential offload

            if is_accelerate_available():
                from accelerate import cpu_offload, disk_offload

                for m in [
                    pipe.vae,  # type: ignore
                    pipe.safety_checker,  # type: ignore
                    pipe.unet,  # type: ignore
                ]:
                    if m is not None:
                        if USE_DISK_OFFLOAD:
                            # If LOW_RAM toggle set (idk why anyone would do this, but it's nice to support stuff
                            # like this in case anyone wants to try running this on fuck knows what)
                            # then offload to disk.
                            disk_offload(
                                m,
                                str(
                                    get_full_model_path(
                                        "offload-dir", model_folder="temp"
                                    )
                                ),
                                device,
                                offload_buffers=True,
                            )
                        else:
                            cpu_offload(m, device, offload_buffers=True)

                logger.info("Optimization: Enabled sequential offload")
            else:
                logger.warning(
                    "Optimization: Sequential offload is not available, because accelerate is not installed"
                )

    if config.api.vae_slicing:
        if not (
            issubclass(pipe.__class__, StableDiffusionUpscalePipeline)
            or isinstance(pipe, StableDiffusionUpscalePipeline)
        ):
            pipe.enable_vae_slicing()
            logger.info("Optimization: Enabled VAE slicing")
        else:
            logger.debug(
                "Optimization: VAE slicing is not available for upscale models"
            )

    if config.api.use_tomesd and not is_for_aitemplate:
        try:
            import tomesd

            tomesd.apply_patch(pipe.unet, ratio=config.api.tomesd_ratio, max_downsample=config.api.tomesd_downsample_layers)  # type: ignore
            logger.info("Optimization: Patched UNet for ToMeSD")
        except ImportError:
            logger.info(
                "Optimization: ToMeSD patch failed, despite having it enabled. Please check installation"
            )

    # pipe.unet = _quantize_module(pipe.unet.to("cpu", dtype=torch.float32), pipe, "cpu", torch.float32)  # type: ignore

    logger.info("Optimization complete")


def _quantize_module(module: torch.nn.Module, pipe: Union[StableDiffusionPipeline, StableDiffusionUpscalePipeline], device: Union[torch.device, str], dtype: torch.dtype) -> torch.nn.Module:
    def setup():
        # Prepare for quantization
        qconfig = torch.ao.quantization.default_qconfig
        float_functional = QFunctional()
        module.qconfig = qconfig
        def setup_quant(_module: torch.nn.Module, number_of_inputs: int = 3, number_of_outputs: int = 1) -> torch.nn.Module:
            setattr(_module, "original_forward_function", _module.forward)  # type: ignore
            for i in range(number_of_inputs):
                setattr(_module, f"quant{i}", torch.ao.quantization.QuantStub(qconfig=qconfig))
            for i in range(number_of_outputs):
                setattr(_module, f"dequant{i}", torch.ao.quantization.DeQuantStub(qconfig=qconfig))
            return _module

        setup_quant(module)
        is_unet = hasattr(module, "time_embedding")
        if is_unet:
            setup_quant(module.time_embedding, 1, 1)  # type: ignore
            setup_quant(module.time_proj, 1, 1)  # type: ignore
        def nte(
            timesteps: torch.Tensor,
            embedding_dim: int,
            flip_sin_to_cos: bool = False,
            downscale_freq_shift: float = 1,
            scale: float = 1,
            max_period: int = 10000,
        ):
            assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

            half_dim = embedding_dim // 2
            exponent = -math.log(max_period) * torch.arange(
                start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
            )
            exponent = exponent / (half_dim - downscale_freq_shift)

            emb = torch.exp(exponent)
            emb = timesteps[:, None] * emb[None, :]

            # scale embeddings
            emb = scale * emb

            # concat sine and cosine embeddings
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

            # flip sine and cosine embeddings
            if flip_sin_to_cos:
                emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

            # zero pad
            if embedding_dim % 2 == 1:
                emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
            return emb

        def new_forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[UNet2DConditionOutput, Tuple]:
            r"""
            Args:
                sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
                timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
                encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
                cross_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                    `self.processor` in
                    [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

            Returns:
                [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
                returning a tuple, the first element is the sample tensor.
            """
            # By default samples have to be AT least a multiple of the overall upsampling factor.
            # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
            # However, the upsampling interpolation output size can be forced to fit any upsampling size
            # on the fly if necessary.
            default_overall_up_factor = 2**self.num_upsamplers

            # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
            forward_upsample_size = False
            upsample_size = None

            if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
                logger.info("Forward upsample size to force interpolation output size.")
                forward_upsample_size = True

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            # 0. center input if necessary
            if self.config.center_input_sample:
                sample = 2 * sample - 1.0

            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None]

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.time_proj(timesteps)

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb

            emb = self.time_embedding(t_emb, timestep_cond)

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)

                if self.config.class_embeddings_concat:
                    emb = torch.cat([emb, class_emb], dim=-1)
                else:
                    emb = emb + class_emb

            if self.time_embed_act is not None:
                emb = self.time_embed_act(emb)

            if self.encoder_hid_proj is not None:
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

            # 2. pre-process
            sample = self.conv_in(sample)

            # 3. down
            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                down_block_res_samples += res_samples

            if down_block_additional_residuals is not None:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
                ):
                    down_block_res_sample = down_block_res_sample + down_block_additional_residual
                    new_down_block_res_samples += (down_block_res_sample,)

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            if mid_block_additional_residual is not None:
                sample = sample + mid_block_additional_residual

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                    )

            # 6. post-process
            if self.conv_norm_out:
                sample = self.conv_norm_out(sample)
                sample = self.conv_act(sample)
            sample = self.conv_out(sample)

            if not return_dict:
                return (sample,)

            return UNet2DConditionOutput(sample=sample)

        emb.get_timestep_embedding = nte

        def new_timestep_embedding_forward(self,
                            sample,
                            condition=False):
            # sample = self.dequant0(sample)  # pylint: disable=no-member type: ignore
            sample = self.original_forward_function(sample, condition)
            return sample
            # return self.quant0(sample)  # pylint: disable=no-member type: ignore

        def new_time_proj_forward(self, timesteps):
            # timesteps = self.dequant0(timesteps)  # pylint: disable=no-member type: ignore
            timesteps = self.original_forward_function(timesteps)
            # timesteps = self.quant0(timesteps)  # pylint: disable=no-member type: ignore
            # def overriden_to(self, dtype=None):
            #     return self
            # timesteps.to = types.MethodType(overriden_to, timesteps)
            return timesteps

        # prepare observers
        module.forward = types.MethodType(new_forward, module)
        if is_unet:
            module.time_embedding.forward = types.MethodType(new_timestep_embedding_forward, module.time_embedding)  # type: ignore
            module.time_proj.forward = types.MethodType(new_time_proj_forward, module.time_proj)  # type: ignore

        
        
        return torch.ao.quantization.prepare(torch.nn.Sequential(torch.ao.quantization.QuantStub(), module, torch.ao.quantization.DeQuantStub()))

    def setup_prompts():
        prompts = ["male Jedi Master portrait by John William Waterhouse and Annie Swynnerton, strong dramatic cinematic lighting. Maxumetric shading photorealistic facial features expressionism intricate detail 8k resolution concept art ornate digital painting illustration gold black red green blue purple teal white pink iridescent colors fantasy atmosphere octane render excellent composition natural textures soft blur bokeh lights matte background trending on Artstation vivid colour palette sharp focus high definition lifelike beautiful",
               "male Jedi Master portrait, art by Artgerm and Greg Rutkowski. highly detailed 8K UHD image of a beautiful attractive young red haired female angel with blue eyes in heaven wearing long intricate ornate robes while animation lightning clouds around her face!! dramatic atmosphere! trending on pixiv fanbox!!! very expressive!!!! expressionly clothed sitting at the top hoof iridescent body armor holding sword made out from flowers!, golden ratio!!!!! character",
               "warrior chief portrait, battle-hardened gaze, tribal markings, weathered skin, fierce expression, traditional attire, feathered headdress, intricate tattoos, sunlit profile, beaded necklace, aged wisdom, ceremonial weapon, piercing eyes, intense atmosphere, commanding presence, powerful stance, ancient symbols, scarred history, vivid storytelling, resolute determination, cultural heritage",
               "woman portrait, captivating gaze, sun-kissed skin, windswept hair, enigmatic smile, golden hour lighting, bokeh background, vintage attire, natural makeup, striking features, artistic pose, vivid colors, soft-focus effect, depth of field, soulful expression, serene ambiance, timeless elegance, chiaroscuro technique, contrast-rich, high-quality portrait, unforgettable aura, masterful composition, ethereal glow"]
        prompts_encoded = []
        for prompt in prompts:
            prompts_encoded.append(pipe._encode_prompt(
                prompt,
                device,
                1,
                True
            ).to(device=device)) # type: ignore
        return prompts_encoded

    prepared = setup()
    prompts_encoded = setup_prompts()[:1]
    num_inference_steps = 6
    batch_size = 1
    height, width = (512, 512)

    with tqdm(total=len(prompts_encoded), unit="prompt", unit_scale=False) as pb:
        for prompt in prompts_encoded:
            pipe.scheduler.set_timesteps(num_inference_steps, device=device)  # type: ignore
            timesteps = pipe.scheduler.timesteps.to(device)  # type: ignore
            latent_timestep = timesteps[:1].repeat(batch_size)  # type: ignore

            # 6. Prepare latent variables
            latents = pipe.prepare_latents(
                latent_timestep,
                batch_size,  # type: ignore
                height,
                width,
                dtype,
                device,
                None,
                None,
            )[0]
            extra_step_kwargs = pipe.prepare_extra_step_kwargs(None, 0.0)
            # 8. Denoising loop
            for _, t in enumerate(pipe.progress_bar(timesteps)):
                noise_pred = prepared(  # type: ignore
                    torch.randn(2, 4, 64, 64), t, prompt
                )[0]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.0 * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(  # type: ignore
                    noise_pred, t, latents, **extra_step_kwargs  # type: ignore
                ).prev_sample  # type: ignore
            pb.update(1)

    del module

    return torch.ao.quantization.convert(prepared)


def send_everything_to_cpu() -> None:
    "Offload module to CPU to save VRAM"

    global gpu_module  # pylint: disable=global-statement

    if gpu_module is not None:
        gpu_module.to("cpu")
    gpu_module = None


def send_to_gpu(module, _) -> None:
    "Load module back to GPU"

    global gpu_module  # pylint: disable=global-statement
    if gpu_module == module:
        return
    if gpu_module is not None:
        gpu_module.to("cpu")
    module.to(_device)
    gpu_module = module


def trace_model(model: torch.nn.Module) -> torch.nn.Module:
    "Traces the model for inference"

    def generate_inputs():
        sample = torch.randn(2, 4, 64, 64).half().cuda()
        timestep = torch.rand(1).half().cuda() * 999
        encoder_hidden_states = torch.randn(2, 77, 768).half().cuda()
        return sample, timestep, encoder_hidden_states

    og = model
    model.eval()
    from functools import partial

    model.forward = partial(model.forward, return_dict=False)
    with torch.inference_mode():
        for _ in range(5):
            model(*generate_inputs())
    model.to(memory_format=torch.channels_last)  # type: ignore
    model = torch.jit.trace(model, generate_inputs(), check_trace=False)  # type: ignore
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(*generate_inputs())

    class TracedUNet(torch.nn.Module):
        "UNet that was JIT traced and should be faster than the original"

        def __init__(self):
            super().__init__()
            self.in_channels = og.in_channels
            self.device = og.device
            self.dtype = og.dtype
            self.config = og.config

        def forward(
            self, latent_model_input, t, encoder_hidden_states
        ) -> UNet2DConditionOutput:
            "Forward pass of the model"

            sample = model(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)

    rn = TracedUNet()
    del og
    return rn


def is_pytorch_pipe(pipe):
    "Checks if the pipe is a pytorch pipe"

    return issubclass(pipe.__class__, (DiffusionPipeline))
