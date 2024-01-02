from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union, List

from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.utils.torch_utils import apply_freeu
import torch

_dummy = None  # here so I can import this


def _unet_new_forward(
    self: UNet2DConditionModel,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int, List[torch.Tensor]],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    quick_replicate: bool = False,  # whether to turn on deepcache
    drop_encode_decode: bool = False,
    replicate_prv_feature: Optional[List[torch.Tensor]] = None,
    cache_layer_id: Optional[int] = None,
    cache_block_id: Optional[int] = None,
    order: Optional[int] = None,
    return_dict: bool = True,
) -> Tuple:
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # 1 - default
    # 2 - deepcache
    # 3 - faster-diffusion
    method = 1
    if quick_replicate:
        method = 2
    elif drop_encode_decode:
        method = 3

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (
            1 - encoder_attention_mask.to(sample.dtype)
        ) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config["center_input_sample"]:
        sample = 2 * sample - 1.0  # type: ignore

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
    elif len(timesteps.shape) == 0:  # type: ignore
        timesteps = timesteps[None].to(sample.device)  # type: ignore

    if len(timesteps.shape) == 1:  # type: ignore
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])  # type: ignore
    t_emb = self.time_proj(timesteps)

    # `Timesteps` does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=sample.dtype)

    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    if self.class_embedding is not None:
        if self.config["class_embed_type"] == "timestep":
            class_labels = self.time_proj(class_labels)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # there might be better ways to encapsulate this.
            class_labels = class_labels.to(dtype=sample.dtype)  # type: ignore

        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

        if self.config["class_embeddings_concat"]:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    if self.config["addition_embed_type"] == "text":
        aug_emb = self.add_embedding(encoder_hidden_states)
    elif self.config["addition_embed_type"] == "text_image":
        # Kandinsky 2.1 - style
        image_embs = added_cond_kwargs.get("image_embeds")  # type: ignore
        text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)  # type: ignore
        aug_emb = self.add_embedding(text_embs, image_embs)
    elif self.config["addition_embed_type"] == "text_time":
        # SDXL - style
        text_embeds = added_cond_kwargs.get("text_embeds")  # type: ignore
        if "time_ids" not in added_cond_kwargs:  # type: ignore
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
            )
        time_ids = added_cond_kwargs.get("time_ids")  # type: ignore
        time_embeds = self.add_time_proj(time_ids.flatten())  # type: ignore
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))  # type: ignore
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)  # type: ignore
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
    elif self.config["addition_embed_type"] == "image":
        # Kandinsky 2.2 - style
        image_embs = added_cond_kwargs.get("image_embeds")  # type: ignore
        aug_emb = self.add_embedding(image_embs)
    elif self.config["addition_embed_type"] == "image_hint":
        # Kandinsky 2.2 - style
        image_embs = added_cond_kwargs.get("image_embeds")  # type: ignore
        hint = added_cond_kwargs.get("hint")  # type: ignore
        aug_emb, hint = self.add_embedding(image_embs, hint)
        sample = torch.cat([sample, hint], dim=1)  # type: ignore

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    if (
        self.encoder_hid_proj is not None
        and self.config["encoder_hid_dim_type"] == "text_proj"
    ):
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
    elif (
        self.encoder_hid_proj is not None
        and self.config["encoder_hid_dim_type"] == "text_image_proj"
    ):
        # Kadinsky 2.1 - style
        image_embeds = added_cond_kwargs.get("image_embeds")  # type: ignore
        encoder_hidden_states = self.encoder_hid_proj(
            encoder_hidden_states, image_embeds
        )
    elif (
        self.encoder_hid_proj is not None
        and self.config["encoder_hid_dim_type"] == "image_proj"
    ):
        # Kandinsky 2.2 - style
        image_embeds = added_cond_kwargs.get("image_embeds")  # type: ignore
        encoder_hidden_states = self.encoder_hid_proj(image_embeds)
    elif (
        self.encoder_hid_proj is not None
        and self.config["encoder_hid_dim_type"] == "ip_image_proj"
    ):
        image_embeds = added_cond_kwargs.get("image_embeds")  # type: ignore
        image_embeds = self.encoder_hid_proj(image_embeds).to(
            encoder_hidden_states.dtype
        )
        encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)

    def downsample(downsample_block, additional_residuals: dict):
        nonlocal sample, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs, encoder_attention_mask, down_intrablock_additional_residuals, down_block_res_samples
        if (
            hasattr(downsample_block, "has_cross_attention")
            and downsample_block.has_cross_attention
        ):
            # For t2i-adapter CrossAttnDownBlock2D
            if is_adapter and len(down_intrablock_additional_residuals) > 0:  # type: ignore
                additional_residuals[
                    "additional_residuals"
                ] = down_intrablock_additional_residuals.pop(  # type: ignore
                    0
                )

            sample, res_samples = downsample_block(
                hidden_states=sample,  # type: ignore
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=emb, scale=1.0
            )
            if is_adapter and len(down_intrablock_additional_residuals) > 0:  # type: ignore
                sample += down_intrablock_additional_residuals.pop(0)  # type: ignore

        down_block_res_samples += res_samples

    prv_f = replicate_prv_feature
    needs_prv = prv_f is None and method == 2

    def upsample(upsample_block, i, length, additional={}):
        nonlocal self, cache_block_id, needs_prv, prv_f, cache_layer_id, down_block_res_samples, forward_upsample_size, sample, emb, encoder_hidden_states, cross_attention_kwargs, upsample_size, attention_mask, encoder_attention_mask

        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[length:]
        down_block_res_samples = down_block_res_samples[:length]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if (
            hasattr(upsample_block, "has_cross_attention")
            and upsample_block.has_cross_attention
        ):
            sample, current_record_f = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                needs_prv=needs_prv,
                **additional,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
                scale=1.0,
            )
            current_record_f = None
        if (
            needs_prv
            and cache_layer_id is not None
            and current_record_f is not None
            and i == len(self.up_blocks) - cache_layer_id - 1
        ):
            assert cache_block_id is not None
            prv_f = current_record_f[-cache_block_id - 1]

    is_controlnet = (
        mid_block_additional_residual is not None
        and down_block_additional_residuals is not None
    )
    is_adapter = down_intrablock_additional_residuals is not None

    if method == 3:
        assert order is not None
        from core.config import config

        mod = config.api.drop_encode_decode

        # ipow = int(np.sqrt(9 + 8 * order))
        cond = order <= 5 or order % 5 == 0
        if isinstance(mod, int):
            # First 5 steps always full cond, just to make sure samples aren't being wasted
            cond = order <= 5 or order % mod == 0

        if cond:
            # 2. pre-process
            sample = self.conv_in(sample)

            # 3. down
            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                downsample(downsample_block, {})

            if is_controlnet:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals  # type: ignore
                ):
                    down_block_res_sample = (
                        down_block_res_sample + down_block_additional_residual
                    )
                    new_down_block_res_samples = new_down_block_res_samples + (
                        down_block_res_sample,
                    )

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:
                if (
                    hasattr(self.mid_block, "has_cross_attention")
                    and self.mid_block.has_cross_attention
                ):
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0  # type: ignore
                    and sample.shape == down_intrablock_additional_residuals[0].shape  # type: ignore
                ):
                    sample += down_intrablock_additional_residuals.pop(0)  # type: ignore

            if is_controlnet:
                sample = sample + mid_block_additional_residual  # type: ignore

            # 4.5. save features
            setattr(self, "skip_feature", deepcopy(down_block_res_samples))
            setattr(self, "toup_feature", sample.detach().clone())
        else:
            down_block_res_samples = self.skip_feature
            sample = self.toup_feature

        for i, upsample_block in enumerate(self.up_blocks):
            upsample(upsample_block, i, -len(upsample_block.resnets))
    else:
        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if (
            cross_attention_kwargs is not None
            and cross_attention_kwargs.get("gligen", None) is not None
        ):
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {
                "objs": self.position_net(**gligen_args)
            }

        # 3. down
        down_block_res_samples = (sample,)
        if method == 1 or (method == 2 and replicate_prv_feature is None):
            for downsample_block in self.down_blocks:
                downsample(downsample_block, {})

            if is_controlnet:
                new_down_block_res_samples = ()

                for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals  # type: ignore
                ):
                    down_block_res_sample = (
                        down_block_res_sample + down_block_additional_residual
                    )
                    new_down_block_res_samples = new_down_block_res_samples + (
                        down_block_res_sample,
                    )

                down_block_res_samples = new_down_block_res_samples

            # 4. mid
            if self.mid_block is not None:
                if (
                    hasattr(self.mid_block, "has_cross_attention")
                    and self.mid_block.has_cross_attention
                ):
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                else:
                    sample = self.mid_block(sample, emb)

                # To support T2I-Adapter-XL
                if (
                    is_adapter
                    and len(down_intrablock_additional_residuals) > 0  # type: ignore
                    and sample.shape == down_intrablock_additional_residuals[0].shape  # type: ignore
                ):
                    sample += down_intrablock_additional_residuals.pop(0)  # type: ignore

            if is_controlnet:
                sample = sample + mid_block_additional_residual  # type: ignore

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                upsample(upsample_block, i, -len(upsample_block.resnets))
        elif method == 2 and replicate_prv_feature is not None:
            assert (
                cache_layer_id is not None
                and cache_block_id is not None
                and replicate_prv_feature is not None
            )
            # Down
            for i, downsample_block in enumerate(self.down_blocks):
                if i > cache_layer_id:
                    break
                downsample(
                    downsample_block,
                    {
                        "exist_block_number": cache_block_id
                        if i == cache_layer_id
                        else None
                    },
                )

            # Skip mid_block

            # Up
            sample = replicate_prv_feature  # type: ignore
            if cache_block_id == len(self.down_blocks[cache_layer_id].attentions):
                cache_block_id = 0
                cache_layer_id += 1
            else:
                cache_block_id += 1

            for i, upsample_block in enumerate(self.up_blocks):
                if i < len(self.up_blocks) - 1 - cache_layer_id:
                    continue

                if i == len(self.up_blocks) - 1 - cache_layer_id:
                    length = cache_block_id + 1
                else:
                    length = len(upsample_block.resnets)

                upsample(
                    upsample_block,
                    i,
                    -length,
                    {
                        "enter_block_number": cache_block_id
                        if i == len(self.up_blocks) - 1 - cache_layer_id
                        else None
                    },
                )
            prv_f = replicate_prv_feature

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)  # type: ignore
    sample = self.conv_out(sample)

    return (
        sample,
        prv_f,
    )


# Changes: added enter_block_number
def _up_new_forward(
    self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
    temb: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    upsample_size: Optional[int] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    enter_block_number: Optional[int] = None,
    needs_prv: bool = False,
) -> Tuple[torch.FloatTensor, List]:
    prv_f = []
    lora_scale = (
        cross_attention_kwargs.get("scale", 1.0)
        if cross_attention_kwargs is not None
        else 1.0
    )
    is_freeu_enabled = (
        getattr(self, "s1", None)
        and getattr(self, "s2", None)
        and getattr(self, "b1", None)
        and getattr(self, "b2", None)
    )

    for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
        if (
            enter_block_number is not None
            and i < len(self.resnets) - enter_block_number - 1
        ):
            continue

        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        # FreeU: Only operate on the first two stages
        if is_freeu_enabled:
            hidden_states, res_hidden_states = apply_freeu(  # type: ignore
                self.resolution_idx,
                hidden_states,
                res_hidden_states,
                s1=self.s1,
                s2=self.s2,
                b1=self.b1,
                b2=self.b2,
            )

        if needs_prv:
            prv_f.append(hidden_states)
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)  # type: ignore

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
            hidden_states = torch.utils.checkpoint.checkpoint(  # type: ignore
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size, scale=lora_scale)

    return hidden_states, prv_f


# Changes: added exist_block_number
def _down_new_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    exist_block_number: Optional[int] = None,
    additional_residuals: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
    output_states = ()

    lora_scale = (
        cross_attention_kwargs.get("scale", 1.0)
        if cross_attention_kwargs is not None
        else 1.0
    )

    blocks = list(zip(self.resnets, self.attentions))

    for i, (resnet, attn) in enumerate(blocks):
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
            hidden_states = torch.utils.checkpoint.checkpoint(  # type: ignore
                create_custom_forward(resnet),
                hidden_states,
                temb,
                **ckpt_kwargs,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
        else:
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        # apply additional residuals to the output of the last pair of resnet and attention blocks
        if i == len(blocks) - 1 and additional_residuals is not None:
            hidden_states = hidden_states + additional_residuals  # type: ignore

        output_states = output_states + (hidden_states,)
        if (
            exist_block_number is not None
            and len(output_states) == exist_block_number + 1
        ):
            return hidden_states, output_states

    if self.downsamplers is not None:
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states, scale=lora_scale)

        output_states = output_states + (hidden_states,)

    return hidden_states, output_states


CrossAttnUpBlock2D.forward = _up_new_forward  # type: ignore
CrossAttnDownBlock2D.forward = _down_new_forward  # type: ignore
UNet2DConditionModel.forward = _unet_new_forward  # type: ignore
