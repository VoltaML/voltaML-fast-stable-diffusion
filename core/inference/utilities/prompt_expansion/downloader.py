from typing import List
from pathlib import Path
import os

from core.config import config
from core.utils import download_file


def _get_download_list() -> List[str]:
    # Another great use for "match," but can't do much about 3.11 not being supported
    model = config.api.prompt_to_prompt_model
    if model == "lllyasviel/Fooocus-Expansion":
        return [
            "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/config.json",
            "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/merges.txt",
            "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/special_tokens_map.json",
            "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/tokenizer.json",
            "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/tokenizer_config.json",
            "https://raw.githubusercontent.com/lllyasviel/Fooocus/main/models/prompt_expansion/fooocus_expansion/vocab.json",
            ("https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin", True),  # type: ignore
        ]
    else:
        return [
            f"https://huggingface.co/{model}/resolve/main/pytorch_model.bin",
            f"https://huggingface.co/{model}/resolve/main/config.json",
            f"https://huggingface.co/{model}/resolve/main/merges.txt",
            f"https://huggingface.co/{model}/resolve/main/special_tokens_map.json",
            f"https://huggingface.co/{model}/resolve/main/tokenizer.json",
            f"https://huggingface.co/{model}/resolve/main/tokenizer_config.json",
            f"https://huggingface.co/{model}/resolve/main/vocab.json",
        ]


def download_model():
    down = _get_download_list()
    model = config.api.prompt_to_prompt_model
    for d in down:
        folder = Path("data/prompt-expansion") / model.split("/")[1]
        if not folder.exists():
            folder.mkdir()
        if isinstance(d, tuple):
            download_file(d[0], folder, add_filename=True)
            os.rename(
                (folder / "fooocus_expansion.bin").absolute().resolve().as_posix(),
                (folder / "pytorch_model.bin").absolute().resolve().as_posix(),
            )
        else:
            download_file(d, folder, add_filename=True)
