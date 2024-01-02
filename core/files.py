import logging
import os
from pathlib import Path
from typing import List, Union

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE as DIFFUSERS_CACHE
from huggingface_hub.file_download import repo_folder_name

from core.types import ModelResponse
from core.utils import determine_model_type

logger = logging.getLogger(__name__)


class CachedModelList:
    "List of models that user has downloaded"

    def __init__(self):
        self.paths = {
            "pytorch": Path(DIFFUSERS_CACHE),
            "checkpoints": Path("data/models"),
            "onnx": Path("data/onnx"),
            "aitemplate": Path("data/aitemplate"),
            "lora": Path("data/lora"),
            "lycoris": Path("data/lycoris"),
            "textual_inversion": Path("data/textual-inversion"),
            "vae": Path("data/vae"),
            "upscaler": Path("data/upscaler"),
            "prompt_expansion": Path("data/prompt-expansion"),
        }

        for _, path in self.paths.items():
            if not path.exists():
                path.mkdir()

        self.ext_whitelist = [".safetensors", ".ckpt", ".pth", ".pt", ".bin"]

    def model_path_to_name(self, path: str) -> str:
        "Return only the stem of a file."

        pth = Path(path)
        return pth.stem

    def pytorch(self) -> List[ModelResponse]:
        "List of models downloaded for PyTorch"

        models: List[ModelResponse] = []

        # Diffusers cached models
        logger.debug(f"Looking for PyTorch models in {self.paths['pytorch']}")
        for model_name in os.listdir(self.paths["pytorch"]):
            logger.debug(f"Found model {model_name}")

            # Skip if it is not a huggingface model
            if "model" not in model_name:
                continue
            parsed_model_name: str = "/".join(model_name.split("--")[1:3])

            try:
                full_path = get_full_model_path(parsed_model_name)
            except ValueError as e:
                logger.debug(f"Model {parsed_model_name} is not valid: {e}")
                continue

            _name, base, stage = determine_model_type(full_path)
            models.append(
                ModelResponse(
                    name=parsed_model_name,
                    path=parsed_model_name,
                    backend="PyTorch",
                    type=base,
                    stage=stage,
                    vae="default",
                    valid=is_valid_diffusers_model(full_path),
                    state="not loaded",
                )
            )

        # Localy stored models
        logger.debug(f"Looking for local models in '{self.paths['checkpoints']}'")
        for model_path in self.paths["checkpoints"].rglob("*"):
            logger.debug(f"Found '{model_path.name}'")

            if model_path.is_dir():
                if not model_path.joinpath("model_index.json").exists():
                    continue

                name, base, stage = determine_model_type(model_path)

                # Assuming that model is in Diffusers format
                models.append(
                    ModelResponse(
                        name=name,
                        path=model_path.relative_to(
                            self.paths["checkpoints"]
                        ).as_posix(),
                        backend="PyTorch",
                        vae="default",
                        valid=is_valid_diffusers_model(model_path),
                        state="not loaded",
                        type=base,
                        stage=stage,
                    )
                )
            elif (
                ".safetensors" in model_path.name or ".ckpt" in model_path.name
            ) and not (
                model_path.parent.joinpath("model_index.json").exists()
                or model_path.parent.parent.joinpath("model_index.json").exists()
            ):
                if ".ckpt" == model_path.suffix:
                    name, base, stage = model_path.name, "SD1.x", "first_stage"
                else:
                    name, base, stage = determine_model_type(model_path)

                # Assuming that model is in Checkpoint / Safetensors format
                models.append(
                    ModelResponse(
                        name=name,
                        path=model_path.relative_to(
                            self.paths["checkpoints"]
                        ).as_posix(),
                        backend="PyTorch",
                        vae="default",
                        valid=True,
                        type=base,
                        stage=stage,
                        state="not loaded",
                    )
                )
            else:
                # Junk file, notify user
                logger.debug(f"Found junk file {model_path}, skipping...")

        return models

    def aitemplate(self) -> List[ModelResponse]:
        "List of models converted to AITempalte"

        models: List[ModelResponse] = []

        logger.debug(f"Looking for AITemplate models in {self.paths['aitemplate']}")

        for model in os.listdir(self.paths["aitemplate"]):
            logger.debug(f"Found model {model}")
            model_name = model.replace("--", "/")

            models.append(
                ModelResponse(
                    name=model_name,
                    path=model,
                    backend="AITemplate",
                    vae="default",
                    valid=is_valid_aitemplate_model(
                        self.paths["aitemplate"].joinpath(model)
                    ),
                    state="not loaded",
                )
            )

        return models

    def onnx(self):
        "List of ONNX models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.paths["onnx"]):
            logger.debug(f"Found ONNX {model}")

            models.append(
                ModelResponse(
                    name=model,
                    path=os.path.join(self.paths["onnx"], model),
                    vae="default",
                    backend="ONNX",
                    valid=True,
                    state="not loaded",
                )
            )
        return models

    def lora(self):
        "List of LoRA models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.paths["lora"]):
            logger.debug(f"Found LoRA {model}")

            # Skip if it is not a LoRA model
            if Path(model).suffix not in self.ext_whitelist:
                continue

            model_name = self.model_path_to_name(model)

            models.append(
                ModelResponse(
                    name=model_name,
                    path=os.path.join(self.paths["lora"], model),
                    vae="default",
                    backend="LoRA",
                    valid=True,
                    state="not loaded",
                )
            )

        return models

    def lycoris(self):
        "List of LyCORIS models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.paths["lycoris"]):
            logger.debug(f"Found LyCORIS {model}")

            # Skip if it is not a LyCORIS model
            if Path(model).suffix not in self.ext_whitelist:
                continue

            model_name = self.model_path_to_name(model)

            models.append(
                ModelResponse(
                    name=model_name,
                    path=os.path.join(self.paths["lycoris"], model),
                    vae="default",
                    backend="LyCORIS",
                    valid=True,
                    state="not loaded",
                )
            )

        return models

    def vae(self):
        "List of VAE models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.paths["vae"]):
            logger.debug(f"Found VAE model {model}")

            path = Path(os.path.join(self.paths["vae"], model))

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
                    state="not loaded",
                )
            )

        return models

    def textual_inversion(self):
        "List of textual inversion models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.paths["textual_inversion"]):
            logger.debug(f"Found textual inversion model {model}")

            # Skip if it is not a Texutal Inversion
            if Path(model).suffix not in self.ext_whitelist:
                continue

            model_name = self.model_path_to_name(model)

            models.append(
                ModelResponse(
                    name=model_name,
                    path=os.path.join(self.paths["textual_inversion"], model),
                    vae="default",
                    backend="Textual Inversion",
                    valid=True,
                    state="not loaded",
                )
            )

        return models

    def upscaler(self):
        "List of upscaler models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.paths["upscaler"]):
            logger.debug(f"Found upscaler model {model}")

            # Skip if it is not an upscaler
            if Path(model).suffix not in self.ext_whitelist:
                continue

            model_name = self.model_path_to_name(model)

            models.append(
                ModelResponse(
                    name=model_name,
                    path=os.path.join(self.paths["upscaler"], model),
                    vae="default",
                    backend="Upscaler",
                    valid=True,
                    state="not loaded",
                )
            )

        return models

    def prompt_expansion(self):
        "List of prompt-expansion (GPT-like) models"

        models: List[ModelResponse] = []

        for model in os.listdir(self.paths["prompt_expansion"]):
            f = os.path.join(self.paths["prompt_expansion"], model)
            if Path(f).is_dir():
                if "config.json" in os.listdir(f):
                    models.append(
                        ModelResponse(model, f, "GPT", True, "", "not loaded", [])
                    )
        return models

    def all(self):
        "List all models"

        return (
            self.pytorch()
            + self.aitemplate()
            + self.onnx()
            + self.lora()
            + self.lycoris()
            + self.textual_inversion()
            + self.vae()
            + self.upscaler()
            + self.prompt_expansion()
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


def current_diffusers_ref(path: str, revision: str = "main") -> str:
    "Return the current ref of the diffusers model"

    rev_path = os.path.join(path, "refs", revision)
    snapshot_path = os.path.join(path, "snapshots")

    if not os.path.exists(rev_path) or not os.path.exists(snapshot_path):
        raise ValueError(
            f"Ref path {rev_path} or snapshot path {snapshot_path} not found"
        )

    snapshots = os.listdir(snapshot_path)

    with open(os.path.join(path, "refs", revision), "r", encoding="utf-8") as f:
        ref = f.read().strip().split(":")[0]

    for snapshot in snapshots:
        if ref.startswith(snapshot):
            return snapshot

    raise ValueError(
        f"Ref {ref} found in {snapshot_path} for revision {revision}, but ref path does not exist"
    )


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
    if alt_path.exists() or force or alt_path.is_symlink():
        logger.debug(f"Found model in {alt_path}")
        return alt_path

    logger.debug(
        f"Model not found in {repo_path} or {alt_path}, checking diffusers cache..."
    )

    # 3. Check if model is stored in diffusers cache
    storage = diffusers_storage_name(repo_id)
    ref = current_diffusers_ref(storage, revision)

    if not ref:
        raise ValueError(f"No ref found for {repo_id}")

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
