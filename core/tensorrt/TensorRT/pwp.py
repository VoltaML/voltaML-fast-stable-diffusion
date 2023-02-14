# Original: https://github.com/ddPn08/Lsmith
# Modified by: Stax124

import re
from typing import List, Union

import numpy as np
import torch
from polygraphy import cuda
from transformers import CLIPTokenizer

from ..utilities import Engine

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
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt(prompt):
    "Parse prompt with multipliers"

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(prompt):
        prompt = m.group(0)
        weight = m.group(1)

        if prompt.startswith("\\"):
            res.append([prompt[1:], 1.0])
        elif prompt == "(":
            round_brackets.append(len(res))
        elif prompt == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif prompt == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif prompt == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([prompt, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def pad_tokens_and_weights(
    tokens, weights, max_length, bos, eos, no_boseos_middle=False, chunk_length=77
):
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = (
        max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    )
    for i in range(len(tokens)):  # pylint: disable=consider-using-enumerate
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


class TensorRTPromptWeightingPipeline:
    "TensorRT pipeline with prompt weighting"

    def __init__(
        self,
        text_encoder: Engine,
        tokenizer: CLIPTokenizer,
        stream: cuda.Stream,
        device=torch.device("cuda"),
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.stream = stream
        self.device = device

    def get_prompts_with_weights(self, prompt: List[str], max_length: int):
        "Get prompt tokens and weights from a list of strings"

        tokens = []
        weights = []
        truncated = False
        for text in prompt:
            texts_and_weights = parse_prompt(text)
            text_token = []
            text_weight = []
            for word, weight in texts_and_weights:
                token = self.tokenizer(
                    word,
                ).input_ids[1:-1]

                text_token += token
                text_weight += [weight] * len(token)
                if len(text_token) > max_length:
                    truncated = True
                    break

            if len(text_token) > max_length:
                truncated = True
                text_token = text_token[:max_length]
                text_weight = text_weight[:max_length]
            tokens.append(text_token)
            weights.append(text_weight)
        if truncated:
            print(
                "Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples"
            )

        return tokens, weights

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = "",
        batch_size: int = 1,
        num_images_per_prompt: int = 1,
        **kwargs,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        max_length = self.tokenizer.model_max_length

        prompt_tokens, prompt_weights = self.get_prompts_with_weights(
            prompt, max_length - 2
        )
        uncond_tokens, uncond_weights = self.get_prompts_with_weights(
            negative_prompt, max_length - 2
        )

        prompt_tokens, prompt_weights = pad_tokens_and_weights(
            prompt_tokens,
            prompt_weights,
            max_length,
            bos,
            eos,
        )
        text_input_ids = torch.tensor(
            prompt_tokens, dtype=torch.int32, device=self.device
        )

        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
        )
        uncond_input_ids = torch.tensor(
            uncond_tokens, dtype=torch.int32, device=self.device
        )

        text_input_ids_inp = cuda.DeviceView(
            ptr=text_input_ids.data_ptr(), shape=text_input_ids.shape, dtype=np.int32
        )
        text_embeddings = self.text_encoder.infer(
            {"input_ids": text_input_ids_inp}, self.stream
        )["text_embeddings"]

        seq_len = text_embeddings.shape[1]
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        max_length = text_input_ids.shape[-1]

        uncond_input_ids_inp = cuda.DeviceView(
            ptr=uncond_input_ids.data_ptr(),
            shape=uncond_input_ids.shape,
            dtype=np.int32,
        )
        uncond_embeddings = self.text_encoder.infer(
            {"input_ids": uncond_input_ids_inp}, self.stream
        )["text_embeddings"]

        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        prompt_weights = torch.tensor(
            prompt_weights, dtype=text_embeddings.dtype, device=self.device
        )
        uncond_weights = torch.tensor(
            uncond_weights,
            dtype=text_embeddings.dtype,
            device=self.device,
        )

        previous_mean = (
            text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        )
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = (
            text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        )
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        previous_mean = (
            uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
        )
        uncond_embeddings *= uncond_weights.unsqueeze(-1)
        current_mean = (
            uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
        )
        uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
