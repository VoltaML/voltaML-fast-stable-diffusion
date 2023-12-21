# A somewhat modified code directly taken from
# https://github.com/lllyasviel/Fooocus/blob/main/modules/expansion.py
#
# Modifications allow different GPT2 models to act as the prompt-to-prompt "expander."

import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from transformers.generation.logits_process import LogitsProcessorList

from core.config import config

from .downloader import download_model

logger = logging.getLogger(__name__)

_seed_limit = 2**32
_magic_split = [
    ", extremely",
    ", intricate,",
    ", very",
    ", ",
]
_dangerous_patterns = "[]【】()（）|:："
_blacklist = [
    "art",
    "digital",
    "Ġpaint",
    "painting",
    "drawing",
    "draw",
    "drawn",
    "concept",
    "illustration",
    "illustrated",
    "illustrate",
    "face",
    "eye",
    "eyes",
    "hand",
    "hands",
    "monster",
    "artistic",
    "oil",
    "brush",
    "artwork",
    "artworks",
]

_CURRENT_MODEL: str = ""
_TOKENIZER: PreTrainedTokenizerBase
_LOGITS_BIAS: torch.Tensor = torch.tensor([0])
_GPT: GPT2PreTrainedModel

_blacklist += ["Ġ" + k for k in _blacklist]


def _safe(string):
    string = str(string)
    for _ in range(16):
        string = string.replace("  ", " ")
    return string.strip(",. \r\n")


def _remove_pattern(string, pattern):
    for p in pattern:
        string = string.replace(p, "")
    return string


def _logits_processor(input_ids: torch.Tensor, scores: torch.Tensor):
    global _LOGITS_BIAS

    _LOGITS_BIAS = _LOGITS_BIAS.to(scores)
    return scores + _LOGITS_BIAS


def _get_directory(prompt_to_prompt_model) -> Path:
    p = Path("data/prompt-expansion/") / prompt_to_prompt_model.split("/")[1]
    if not p.exists():
        download_model()
    return p


def _device_dtype(prompt_to_prompt_device) -> Tuple[torch.device, torch.dtype]:
    device = (
        torch.device(config.api.device)
        if prompt_to_prompt_device == "gpu"
        else torch.device("cpu")
    )
    dtype = config.api.load_dtype if prompt_to_prompt_device == "gpu" else torch.float32
    return (device, dtype)


def _load(prompt_to_prompt_model, prompt_to_prompt_device):
    global _CURRENT_MODEL, _LOGITS_BIAS, _GPT, _TOKENIZER

    if _CURRENT_MODEL != prompt_to_prompt_model:
        _CURRENT_MODEL = prompt_to_prompt_model

        # Setup tokenizer and logits bias according to:
        # - https://huggingface.co/blog/introducing-csearch
        # - https://huggingface.co/docs/transformers/generation_strategies
        _TOKENIZER = AutoTokenizer.from_pretrained(
            _get_directory(prompt_to_prompt_model)
        )
        vocab: dict = _TOKENIZER.vocab  # type: ignore
        _LOGITS_BIAS = torch.zeros((1, len(vocab)), dtype=torch.float32)
        _LOGITS_BIAS[0, _TOKENIZER.eos_token_id] = -16.0
        _LOGITS_BIAS[0, 198] = -1024.0
        for k, v in vocab.items():
            if k in _blacklist:
                _LOGITS_BIAS[0, v] = -1024.0

        _GPT = AutoModelForCausalLM.from_pretrained(
            _get_directory(prompt_to_prompt_model)
        )
        _GPT.eval()

    device, dtype = _device_dtype(prompt_to_prompt_device)
    _GPT = _GPT.to(device=device, dtype=dtype)  # type: ignore


@torch.inference_mode()
def expand(prompt, seed, prompt_expansion_settings: Dict):
    if prompt == "":
        return ""

    prompt_to_prompt_model = prompt_expansion_settings.pop(
        "prompt_to_prompt_model", config.api.prompt_to_prompt_model
    )
    prompt_to_prompt_device = prompt_expansion_settings.pop(
        "prompt_to_prompt_device", config.api.prompt_to_prompt_device
    )

    # Load in the model, or move to the necessary device.
    _load(prompt_to_prompt_model, prompt_to_prompt_device)

    seed = int(seed) % _seed_limit
    set_seed(seed)
    logger.debug(f"Using seed {seed}")
    origin = _safe(prompt)
    prompt = origin + _magic_split[seed % len(_magic_split)]

    device, _ = _device_dtype(prompt_to_prompt_device)

    tokenized_kwargs = _TOKENIZER(prompt, return_tensors="pt")
    tokenized_kwargs.data["input_ids"] = tokenized_kwargs.data["input_ids"].to(
        device=device
    )
    tokenized_kwargs.data["attention_mask"] = tokenized_kwargs.data[
        "attention_mask"
    ].to(device=device)

    current_token_length = int(tokenized_kwargs.data["input_ids"].shape[1])
    max_token_length = 75 * int(math.ceil(float(current_token_length) / 75.0))
    max_new_tokens = max_token_length - current_token_length

    logits_processor = LogitsProcessorList([_logits_processor])

    features = _GPT.generate(
        **tokenized_kwargs,  # type: ignore
        num_beams=1,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        logits_processor=logits_processor,
    )

    result = _TOKENIZER.batch_decode(features, skip_special_tokens=True)
    result = result[0]
    result = _safe(result)
    result = _remove_pattern(result, _dangerous_patterns)
    return result
