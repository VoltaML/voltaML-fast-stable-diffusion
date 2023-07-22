import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from diffusers.utils.constants import DIFFUSERS_CACHE
from huggingface_hub.file_download import repo_folder_name

from core.types import ModelResponse

logger = logging.getLogger(__name__)


class CachedModelList:
    "List of models downloaded for PyTorch and (or) converted to TRT"

    def __init__(self):
        self.pytorch_path = Path(DIFFUSERS_CACHE)
        self.checkpoint_converted_path = Path("data/models")
        self.onnx_path = Path("data/onnx")
        self.aitemplate_path = Path("data/aitemplate")
        self.lora_path = Path("data/lora")
        self.textual_inversion_path = Path("data/textual-inversion")
        self.vae_path = Path("data/vae")

        self.ext_whitelist = [".safetensors", ".ckpt", ".pth", ".pt", ".bin"]

    def model_path_to_name(self, path: str) -> str:
        "Return only the stem of a file."

        pth = Path(path)
        return pth.stem

    def pytorch(self) -> List[ModelResponse]:
        "List of models downloaded for PyTorch"

        models: List[ModelResponse] = []

        # Diffusers cached models
        logger.debug(f"Looking for PyTorch models in {self.pytorch_path}")
        for model_name in os.listdir(self.pytorch_path):
            logger.debug(f"Found model {model_name}")

            # Skip if it is not a huggingface model
            if not "model" in model_name:
                continue

            name: str = "/".join(model_name.split("--")[1:3])
            try:
                file = get_full_model_path(name) / "model_index.json"
                with open(file, "r", encoding="UTF-8") as content:
                    sdxl = "StableDiffusionXL" in content.readlines()[1]
                    if sdxl:
                        models.append(
                            ModelResponse(
                                name=model_name,
                                path=model_name,
                                backend="SDXL",
                                vae="default",
                                valid=True,
                                loras=[],
                                state="not loaded",
                            )
                        )
                        continue
            except Exception:  # pylint: disable=broad-except
                pass
            try:
                models.append(
                    ModelResponse(
                        name=name,
                        path=name,
                        backend="PyTorch",
                        vae="default",
                        valid=is_valid_diffusers_model(get_full_model_path(name)),
                        loras=[],
                        state="not loaded",
                    )
                )
            except ValueError:
                logger.debug(f"Invalid model {name}, skipping...")
                continue

        # Localy stored models
        logger.debug(f"Looking for local models in {self.checkpoint_converted_path}")
        for model_name in os.listdir(self.checkpoint_converted_path):
            logger.debug(f"Found model {model_name}")

            if self.checkpoint_converted_path.joinpath(model_name).is_dir():
                # Assuming that model is in Diffusers format
                file = self.checkpoint_converted_path / model_name / "model_index.json"
                try:
                    with open(file, "r", encoding="UTF-8") as content:
                        sdxl = "StableDiffusionXL" in content.readlines()[1]
                        if sdxl:
                            models.append(
                                ModelResponse(
                                    name=model_name,
                                    path=model_name,
                                    backend="SDXL",
                                    vae="default",
                                    valid=True,
                                    loras=[],
                                    state="not loaded",
                                )
                            )
                            continue
                except Exception:  # pylint: disable=broad-except
                    pass
                models.append(
                    ModelResponse(
                        name=model_name,
                        path=model_name,
                        backend="PyTorch",
                        vae="default",
                        valid=is_valid_diffusers_model(
                            self.checkpoint_converted_path.joinpath(model_name)
                        ),
                        loras=[],
                        state="not loaded",
                    )
                )
            elif (self.checkpoint_converted_path / model_name).suffix in [
                ".ckpt",
                ".safetensors",
            ]:
                # Assuming that model is in Checkpoint / Safetensors format
                models.append(
                    ModelResponse(
                        name=model_name,
                        path=model_name,
                        backend="PyTorch"
                        if "xl" not in model_name.casefold()
                        else "SDXL",
                        vae="default",
                        valid=True,
                        loras=[],
                        state="not loaded",
                    )
                )
            else:
                # Junk file, notify user
                logger.debug(
                    f"Found junk file {model_name} in {self.checkpoint_converted_path}, skipping..."
                )

        return models

    def aitemplate(self) -> List[ModelResponse]:
        "List of models converted to TRT"

        models: List[ModelResponse] = []

        logger.debug(f"Looking for AITemplate models in {self.aitemplate_path}")

        for model in os.listdir(self.aitemplate_path):
            logger.debug(f"Found model {model}")
            model_name = model.replace("--", "/")

            models.append(
                ModelResponse(
                    name=model_name,
                    path=model,
                    backend="AITemplate",
                    vae="default",
                    valid=is_valid_aitemplate_model(
                        self.aitemplate_path.joinpath(model)
                    ),
                    loras=[],
                    state="not loaded",
                )
            )

        return models

    def onnx(self):
        "List of ONNX models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.onnx_path):
            logger.debug(f"Found ONNX {model}")

            models.append(
                ModelResponse(
                    name=model,
                    path=os.path.join(self.onnx_path, model),
                    vae="default",
                    backend="ONNX",
                    valid=True,
                    loras=[],
                    state="not loaded",
                )
            )
        return models

    def lora(self):
        "List of LoRA models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.lora_path):
            logger.debug(f"Found LoRA {model}")

            # Skip if it is not a LoRA model
            if not any(x in model for x in self.ext_whitelist):
                continue

            model_name = self.model_path_to_name(model)

            models.append(
                ModelResponse(
                    name=model_name,
                    path=os.path.join(self.lora_path, model),
                    vae="default",
                    backend="LoRA",
                    valid=True,
                    loras=[],
                    state="not loaded",
                )
            )

        return models

    def vae(self):
        "List of VAE models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.vae_path):
            logger.debug(f"Found VAE model {model}")

            path = Path(os.path.join(self.vae_path, model))

            # Skip if it is not a VAE model
            if path.suffix not in self.ext_whitelist and not path.is_dir():
                continue
            if path.is_dir() and not is_valid_diffusers_vae(path):
                continue

            models.append(
                ModelResponse(
                    name=path.stem,
                    path=path.as_posix(),
                    backend="VAE",
                    valid=True,
                    vae="default",
                    loras=[],
                    state="not loaded",
                )
            )

        return models

    def textual_inversion(self):
        "List of textual inversion models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.textual_inversion_path):
            logger.debug(f"Found textual inversion model {model}")

            # Skip if it is not a Texutal Inversion
            if not any(x in model for x in self.ext_whitelist):
                continue

            model_name = self.model_path_to_name(model)

            models.append(
                ModelResponse(
                    name=model_name,
                    path=os.path.join(self.textual_inversion_path, model),
                    vae="default",
                    backend="Textual Inversion",
                    valid=True,
                    loras=[],
                    state="not loaded",
                )
            )

        return models

    def all(self):
        "List all models"

        return (
            self.pytorch()
            + self.aitemplate()
            + self.onnx()
            + self.lora()
            + self.textual_inversion()
            + self.vae()
        )


def is_valid_diffusers_vae(model_path: Path) -> bool:
    "Check if the folder contains valid VAE files."

    files = ["config.json", "diffusion_pytorch_model.bin"]
    is_valid = True
    for file in files:
        is_valid = is_valid and Path(os.path.join(model_path, file)).exists()
    return is_valid


def is_valid_diffusers_model(model_path: Union[str, Path]):
    "Check if the folder contains valid diffusers files"

    binary_folders = ["text_encoder", "unet", "vae"]

    files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "unet/config.json",
        "vae/config.json",
    ]

    is_valid = True

    path = model_path if isinstance(model_path, Path) else Path(model_path)

    # Check all the folders that should container at least one binary file
    for folder in binary_folders:
        # Check if the folder exists
        if not os.path.exists(path / folder):
            is_valid = False
            break

        # Check if there is at least one .bin file in the folder
        has_binaries = True
        found_files = os.listdir(path / folder)
        if len([path / folder / i for i in found_files if i.endswith(".bin")]) < 1:
            has_binaries = False

        # Check if there is at least one .safetensors file in the folder
        has_safetensors = True
        found_files = os.listdir(path / folder)
        if (
            len([path / folder / i for i in found_files if i.endswith(".safetensors")])
            < 1
        ):
            has_safetensors = False

        # If there is no binary or safetensor file, the model is not valid
        if not has_binaries and not has_safetensors:
            is_valid = False

    # Check all the other files that should be present
    for file in files:
        if not os.path.exists(path / file):
            is_valid = False

    return is_valid


def is_valid_aitemplate_model(model_path: Union[str, Path]):
    "Check if the folder contains valid AITemplate files"

    files = [
        "AutoencoderKL/test.so",
        "CLIPTextModel/test.so",
        "UNet2DConditionModel/test.so",
    ]

    is_valid = True
    path = model_path if isinstance(model_path, Path) else Path(model_path)

    # Check all the files that should be present
    for file in files:
        if not os.path.exists(path / file):
            is_valid = False

    return is_valid


def diffusers_storage_name(repo_id: str, repo_type: str = "model") -> str:
    "Return the name of the folder where the diffusers model is stored"

    return os.path.join(
        DIFFUSERS_CACHE, repo_folder_name(repo_id=repo_id, repo_type=repo_type)
    )


def current_diffusers_ref(path: str, revision: str = "main") -> Optional[str]:
    "Return the current ref of the diffusers model"

    if not os.path.exists(os.path.join(path, "refs", revision)):
        return None

    snapshots = os.listdir(os.path.join(path, "snapshots"))
    ref = ""

    with open(os.path.join(path, "refs", revision), "r", encoding="utf-8") as f:
        ref = f.read().strip().split(":")[0]

    for snapshot in snapshots:
        if ref.startswith(snapshot):
            return snapshot


def get_full_model_path(
    repo_id: str,
    revision: str = "main",
    model_folder: str = "models",
    force: bool = False,
    diffusers_skip_ref_follow: bool = False,
) -> Path:
    "Return the path to the actual model"

    # Replace -- with / and remove the __dim part
    repo_id = repo_id.replace("--", "/").split("__")[0]
    repo_path = Path(repo_id)

    # 1. Check for the exact path
    if repo_path.exists():
        logger.debug(f"Found model in {repo_path}")
        return repo_path

    # 2. Check if model is stored in local storage
    alt_path = Path("data") / model_folder / repo_id
    if alt_path.exists() or force:
        logger.debug(f"Found model in {alt_path}")
        return alt_path

    logger.debug(
        f"Model not found in {repo_path} or {alt_path}, checking diffusers cache..."
    )

    # 3. Check if model is stored in diffusers cache
    storage = diffusers_storage_name(repo_id)
    ref = current_diffusers_ref(storage, revision)

    if not ref:
        raise ValueError("No ref found")

    if diffusers_skip_ref_follow:
        return Path(storage)

    return Path(storage) / "snapshots" / ref


def list_cached_model_folders(repo_id: str, full_path: bool = False):
    "List all the folders in the cached model"

    storage = diffusers_storage_name(repo_id)
    ref = current_diffusers_ref(storage)

    if full_path:
        return [
            os.path.join(storage, "snapshots", ref, folder)  # type: ignore
            for folder in os.listdir(os.path.join(storage, "snapshots", ref))  # type: ignore
        ]
    else:
        return os.listdir(os.path.join(storage, "snapshots", ref))  # type: ignore
