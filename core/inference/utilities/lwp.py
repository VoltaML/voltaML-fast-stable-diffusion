import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from core.utils import download_file

from ...config import config
from ...files import get_full_model_path
from .prompt_expansion import expand

logger = logging.getLogger(__name__)

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:[\s]*([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


class Placebo:
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    loras: list


special_parser = re.compile(
    r"\<(lora|ti):([^\:\(\)\<\>\[\]]+)(?::[\s]*([+-]?(?:[0-9]*[.])?[0-9]+))?\>|\<(lora|ti):(http[^\(\)\<\>\[\]]+\/[^:]+)(?::[\s]*([+-]?(?:[0-9]*[.])?[0-9]+))?\>"
)


def parse_prompt_special(
    text: str,
) -> Tuple[str, Dict[str, List[Union[str, Tuple[str, float]]]]]:
    """
    Replaces special tokens like <lora:more_details:0.7> with their correct format and outputs then into a dict.

    >>> parse_prompt_special("This is a <ti:easynegative:0.7> example.")
    'This is a (easynegative:0.7) example.'
    >>> parse_prompt_special("This is a <lora:more_details:0.7> example.")
    'This is a  example.' (lora "more_details" gets loaded with a=0.7)
    """

    load_map = {}

    def replace(match):
        type_: str = match.group(4) or match.group(1)
        name = match.group(2)
        strength = match.group(6) or match.group(3)
        url: str = match.group(5)

        if url:
            filename = url.split("/")[-1]
            file = Path("data/lora") / filename
            name = file.stem

            # Check if file exists
            if not file.exists():
                name = download_file(url, Path("data/lora"), add_filename=True).stem
            else:
                logger.debug(f"File {file} already cached")

        load_map[type_] = load_map.get(type_, [])
        if type_ == "ti":
            load_map[type_].append(name)
            return f"({name}:{strength if strength else '1.0'})"
        # LoRAs don't really have trigger words, they all modify the UNet at least a bit
        load_map[type_].append((name, float(strength) if strength else 1.0))
        return "" if not config.api.huggingface_style_parsing else name

    parsed = special_parser.sub(replace, text)
    return (parsed, load_map)


def parse_prompt_attention(text):
    r"""
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    logger.debug(f"Weighted prompt: {res}")

    return res


def get_prompts_with_weights(tokenizer, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.

    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer(word, max_length=max_length, truncation=True).input_ids[1:-1]  # type: ignore
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning(
            "Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples"
        )
    return tokens, weights


def pad_tokens_and_weights(
    tokens, weights, max_length, bos, eos, no_boseos_middle=True, chunk_length=77
):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = (
        max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    )
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][
                        j
                        * (chunk_length - 2) : min(
                            len(weights[i]), (j + 1) * (chunk_length - 2)
                        )
                    ]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe: StableDiffusionPipeline,
    text_input: torch.Tensor,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
    text_encoder=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """

    # TODO: when SDXL releases, refactor CLIP_stop_at_last_layer here.

    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)

    if not hasattr(pipe, "text_encoder_2"):
        if max_embeddings_multiples > 1:
            text_embeddings = []
            for i in range(max_embeddings_multiples):
                # extract the i-th chunk
                text_input_chunk = text_input[
                    :, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2
                ].clone()

                # cover the head and the tail by the starting and the ending tokens
                text_input_chunk[:, 0] = text_input[0, 0]
                text_input_chunk[:, -1] = text_input[0, -1]
                if hasattr(pipe, "clip_inference"):
                    text_embedding = pipe.clip_inference(text_input_chunk)
                else:
                    text_embedding = pipe.text_encoder(text_input_chunk)[0]  # type: ignore

                if no_boseos_middle:
                    if i == 0:
                        # discard the ending token
                        text_embedding = text_embedding[:, :-1]
                    elif i == max_embeddings_multiples - 1:
                        # discard the starting token
                        text_embedding = text_embedding[:, 1:]
                    else:
                        # discard both starting and ending tokens
                        text_embedding = text_embedding[:, 1:-1]

                text_embeddings.append(text_embedding)
            text_embeddings = torch.concat(text_embeddings, axis=1)  # type: ignore
        else:
            if hasattr(pipe, "clip_inference"):
                text_embeddings = pipe.clip_inference(text_input)
            else:
                text_embeddings = pipe.text_encoder(text_input)[0]  # type: ignore
        return text_embeddings, None  # type: ignore
    else:
        if max_embeddings_multiples > 1:
            text_embeddings = []
            hidden_states = []
            for i in range(max_embeddings_multiples):
                text_input_chunk = text_input[
                    :, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2
                ].clone()

                text_input_chunk[:, 0] = text_input[0, 0]
                text_input_chunk[:, -1] = text_input[0, -1]
                text_embedding = text_encoder(  # type: ignore
                    text_input_chunk, output_hidden_states=True
                )

                if no_boseos_middle:
                    if i == 0:
                        text_embedding.hidden_states[-2] = text_embedding.hidden_states[
                            -2
                        ][:, :-1]
                    elif i == max_embeddings_multiples - 1:
                        text_embedding.hidden_states[-2] = text_embedding.hidden_states[
                            -2
                        ][:, 1:]
                    else:
                        text_embedding.hidden_states[-2] = text_embedding.hidden_states[
                            -2
                        ][:, 1:-1]
                text_embeddings.append(text_embedding)
            text_embeddings = torch.concat([x.hidden_states[-2] for x in text_embeddings], axis=1)  # type: ignore
            # Temporary, but hey, at least it works :)
            # TODO: try and fix this monstrosity :/
            hidden_states = text_embeddings[-1][0].unsqueeze(0)  # type: ignore
            # text_embeddings = torch.Tensor(hidden_states.shape[0])
        else:
            text_embeddings = text_encoder(text_input, output_hidden_states=True)  # type: ignore
            hidden_states = text_embeddings[0]
            text_embeddings = text_embeddings.hidden_states[-2]
        logger.debug(f"{hidden_states.shape} {text_embeddings.shape}")
        return text_embeddings, hidden_states


def get_weighted_text_embeddings(
    pipe: StableDiffusionPipeline,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    seed: int = -1,
    prompt_expansion_settings: Optional[Dict] = None,
    text_encoder=None,
    tokenizer=None,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.

    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.

    Args:
        pipe (`StableDiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `3`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    prompt_expansion_settings = prompt_expansion_settings or {}

    tokenizer = tokenizer or pipe.tokenizer
    text_encoder = text_encoder or pipe.text_encoder

    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2  # type: ignore
    if isinstance(prompt, str):
        prompt = [prompt]

    if not hasattr(pipe, "clip_inference"):
        loralist = []
        for i, p in enumerate(prompt):
            prompt[i], load_map = parse_prompt_special(p)
            if len(load_map.keys()) != 0:
                logger.debug(load_map)
                if "lora" in load_map:
                    from ..injectables import install_lora_hook

                    install_lora_hook(pipe)

                    for lora, alpha in load_map["lora"]:
                        correct_path = get_full_model_path(
                            lora, model_folder="lora", force=True
                        )
                        for ext in [".safetensors", ".ckpt", ".bin", ".pt", ".pth"]:
                            path = get_full_model_path(
                                lora + ext, model_folder="lora", force=True
                            )
                            if path.exists():
                                correct_path = path
                                break
                        if not correct_path.exists():
                            correct_path = get_full_model_path(
                                lora, model_folder="lycoris", force=True
                            )
                            for ext in [".safetensors", ".ckpt", ".bin", ".pt", ".pth"]:
                                path = get_full_model_path(
                                    lora + ext, model_folder="lycoris", force=True
                                )
                                if path.exists():
                                    correct_path = path
                                    break
                            if correct_path.exists():
                                loralist.append((correct_path, alpha, "lycoris"))
                            else:
                                logger.error(
                                    f"Could not find any lora with the name {lora}"
                                )
                        else:
                            loralist.append((correct_path, alpha))
                if "ti" in load_map:
                    # Disable TI for now as there's no reliable way to unload them
                    logger.info(
                        "Textual inversion via prompts is temporarily disabled."
                    )

        if config.api.huggingface_style_parsing and hasattr(pipe, "lora_injector"):
            for prompt_ in prompt:
                old_l = loralist
                loralist = list(
                    filter(
                        lambda x: Path(x[0]).stem.casefold() in prompt_.casefold(),
                        loralist,
                    )
                )
                for lora, weight in old_l:
                    if (lora, weight) not in loralist:
                        try:
                            pipe.remove_lora(Path(lora).name)
                            logger.debug(f"Unloading LoRA: {Path(lora).name}")
                        except KeyError:
                            pass
        if hasattr(pipe, "lora_injector"):
            remove_loras = pipe.unload_loras
            remove_lycoris = pipe.unload_lycoris
            logger.debug(f"{loralist}")
            for ent in loralist:
                lyco = len(ent) == 3
                lora = ent[0]
                alpha = ent[1]
                name = Path(lora).name

                if lyco:
                    if name not in remove_lycoris:
                        logger.debug(f"Adding LyCORIS {name} to the removal list")
                        remove_lycoris.append(name)
                    logger.debug(f"Applying LyCORIS {name} with strength {alpha}")
                    pipe.lora_injector.apply_lycoris(lora, alpha)
                else:
                    if name not in remove_loras:
                        logger.debug(f"Adding LoRA {name} to the removal list")
                        remove_loras.append(name)
                    logger.debug(f"Applying LoRA {name} with strength {alpha}")
                    pipe.lora_injector.apply_lora(lora, alpha)
            pipe.unload_lycoris = remove_lycoris
            pipe.unload_loras = remove_loras

    # Move after loras to purge <lora:...> and <ti:...>
    if prompt_expansion_settings.pop("prompt_to_prompt", config.api.prompt_to_prompt):
        for i, p in enumerate(prompt):
            prompt[i] = expand(
                p, seed, prompt_expansion_settings=prompt_expansion_settings
            )
            logger.info(f'Expanded prompt to "{prompt[i]}"')

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(
            tokenizer, prompt, max_length - 2
        )
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(
                tokenizer, uncond_prompt, max_length - 2
            )
    else:
        prompt_tokens = [
            token[1:-1]
            for token in tokenizer(  # type: ignore
                prompt, max_length=max_length, truncation=True
            ).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1]
                for token in tokenizer(  # type: ignore
                    uncond_prompt, max_length=max_length, truncation=True
                ).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))  # type: ignore

    max_embeddings_multiples = min(
        max_embeddings_multiples,  # type: ignore
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,  # type: ignore
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)  # type: ignore
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2  # type: ignore

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id  # type: ignore
    eos = tokenizer.eos_token_id  # type: ignore
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,  # type: ignore
        chunk_length=tokenizer.model_max_length,  # type: ignore
    )
    prompt_tokens = torch.tensor(
        prompt_tokens, dtype=torch.long, device=pipe.device if hasattr(pipe, "clip_inference") else text_encoder.device  # type: ignore
    )
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,  # type: ignore
            uncond_weights,  # type: ignore
            max_length,
            bos,
            eos,
            no_boseos_middle=no_boseos_middle,  # type: ignore
            chunk_length=tokenizer.model_max_length,  # type: ignore
        )
        uncond_tokens = torch.tensor(
            uncond_tokens, dtype=torch.long, device=pipe.device if hasattr(pipe, "clip_inference") else text_encoder.device  # type: ignore
        )

    # get the embeddings
    text_embeddings, hidden_states = get_unweighted_text_embeddings(
        pipe,  # type: ignore
        prompt_tokens,
        tokenizer.model_max_length,  # type: ignore
        no_boseos_middle=no_boseos_middle,
        text_encoder=text_encoder,
    )
    prompt_weights = torch.tensor(
        prompt_weights, dtype=text_embeddings.dtype, device=pipe.device if hasattr(pipe, "clip_inference") else text_encoder.device  # type: ignore
    )
    if uncond_prompt is not None:
        uncond_embeddings, uncond_hidden_states = get_unweighted_text_embeddings(
            pipe,  # type: ignore
            uncond_tokens,  # type: ignore
            tokenizer.model_max_length,  # type: ignore
            no_boseos_middle=no_boseos_middle,
            text_encoder=text_encoder,
        )
        uncond_weights = torch.tensor(
            uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device if hasattr(pipe, "clip_inference") else text_encoder.device  # type: ignore
        )

    # assign weights to the prompts and normalize in the sense of mean
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = (
            text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)  # type: ignore
        )
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = (
            text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)  # type: ignore
        )
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = (
                uncond_embeddings.float()  # type: ignore
                .mean(axis=[-2, -1])
                .to(uncond_embeddings.dtype)  # type: ignore
            )
            uncond_embeddings *= uncond_weights.unsqueeze(-1)  # type: ignore
            current_mean = (
                uncond_embeddings.float()  # type: ignore
                .mean(axis=[-2, -1])  # type: ignore
                .to(uncond_embeddings.dtype)
            )
            uncond_embeddings *= (
                (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
            )

    if uncond_prompt is not None:
        return text_embeddings, hidden_states, uncond_embeddings, uncond_hidden_states  # type: ignore
    return text_embeddings, hidden_states, None, None
