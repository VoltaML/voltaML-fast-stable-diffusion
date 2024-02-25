from copy import deepcopy

from diffusers.loaders import (
    TextualInversionLoaderMixin,
    load_textual_inversion_state_dicts,
)
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


def maybe_convert_prompt(prompt: str, tokenizer: PreTrainedTokenizer) -> str:
    tokens = tokenizer.tokenize(prompt)
    unique_tokens = set(tokens)
    for token in unique_tokens:
        if token in tokenizer.added_tokens_encoder:
            replacement = token
            i = 1
            while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                replacement += f" {token}_{i}"
                i += 1
            prompt = prompt.replace(token, replacement)
    return prompt


def unload(token: str, tokenizer: PreTrainedTokenizer, text_encoder: PreTrainedModel):
    load_map = text_encoder.change_map if hasattr(text_encoder, "change_map") else []
    input_embedding: torch.Tensor = text_encoder.get_input_embeddings().weight
    device, dtype = text_encoder.device, text_encoder.dtype

    if token in load_map:
        token_id: int = tokenizer.convert_tokens_to_ids(token)  # type: ignore
        tokenizer.added_tokens_encoder.pop(token)
        input_embedding.data = torch.cat(
            (input_embedding.data[:token_id], input_embedding.data[token_id + 1 :])
        )
        text_encoder.resize_token_embeddings(len(tokenizer))
        load_map.remove(token)

    input_embedding.to(device, dtype)
    setattr(text_encoder, "change_map", load_map)


def unload_all(tokenizer: PreTrainedTokenizer, text_encoder: PreTrainedModel):
    load_map = text_encoder.change_map if hasattr(text_encoder, "change_map") else []
    input_embedding: torch.Tensor = text_encoder.get_input_embeddings().weight
    device, dtype = text_encoder.device, text_encoder.dtype

    for token in deepcopy(load_map):
        token_id: int = tokenizer.convert_tokens_to_ids(token)  # type: ignore
        tokenizer.added_tokens_encoder.pop(token)
        input_embedding.data = torch.cat(
            (input_embedding.data[:token_id], input_embedding.data[token_id + 1 :])
        )
        text_encoder.resize_token_embeddings(len(tokenizer))
        load_map.remove(token)

    input_embedding.to(device, dtype)
    setattr(text_encoder, "change_map", load_map)


def load(
    model: str,
    token: str,
    tokenizer: PreTrainedTokenizer,
    text_encoder: PreTrainedModel,
):
    state_dicts = load_textual_inversion_state_dicts(model)

    token, embeddings = TextualInversionLoaderMixin._retrieve_tokens_and_embeddings(
        [token],
        state_dicts,
        tokenizer,  # type: ignore
    )
    tokens, embeddings = TextualInversionLoaderMixin._retrieve_tokens_and_embeddings(
        token, embeddings, tokenizer
    )

    device, dtype = text_encoder.device, text_encoder.dtype

    load_map = text_encoder.change_map if hasattr(text_encoder, "change_map") else []
    input_embedding: torch.Tensor = text_encoder.get_input_embeddings().weight

    def load(token, embedding):
        tokenizer.add_tokens(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        input_embedding.data[token_id] = embedding
        text_encoder.resize_token_embeddings(len(tokenizer))
        load_map.append(token)

    for _token, embedding in zip(tokens, embeddings):
        load(_token, embedding)

    input_embedding.to(device, dtype)
    setattr(text_encoder, "change_map", load_map)
