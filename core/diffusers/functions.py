import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import (
    CONFIG_NAME,
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    ONNX_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from huggingface_hub import model_info  # type: ignore
from huggingface_hub._snapshot_download import snapshot_download
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import ModelInfo
from huggingface_hub.utils._errors import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests import HTTPError

logger = logging.getLogger(__name__)
config_name = "model_index.json"


def load_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    return_unused_kwargs=False,
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    r"""
    Instantiate a Python class from a config dictionary

    Parameters:
        pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
            Can be either:

                - A string, the *model id* of a model repo on huggingface.co. Valid model ids should have an
                  organization name, like `google/ddpm-celebahq-256`.
                - A path to a *directory* containing model weights saved using [`~ConfigMixin.save_config`], e.g.,
                  `./my_model_directory/`.

        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received files. Will attempt to resume the download if such a
            file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        output_loading_info(`bool`, *optional*, defaults to `False`):
            Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
        local_files_only(`bool`, *optional*, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo (either remote in
            huggingface.co or downloaded locally), you can specify the folder name here.

    <Tip>

     It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
     models](https://huggingface.co/docs/hub/models-gated#gated-models).

    </Tip>

    <Tip>

    Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
    use this method in a firewalled environment.

    </Tip>
    """
    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    use_auth_token = kwargs.pop("use_auth_token", None)
    local_files_only = kwargs.pop("local_files_only", False)
    revision = kwargs.pop("revision", None)
    _ = kwargs.pop("mirror", None)
    subfolder = kwargs.pop("subfolder", None)

    user_agent = {"file_type": "config"}

    pretrained_model_name_or_path = str(pretrained_model_name_or_path)

    if config_name is None:
        raise ValueError(
            "`self.config_name` is not defined. Note that one should not load a config from "
            "`ConfigMixin`. Please make sure to define `config_name` in a class inheriting from `ConfigMixin`"
        )

    if os.path.isfile(pretrained_model_name_or_path):
        config_file = pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, config_name)):
            # Load from a PyTorch checkpoint
            config_file = os.path.join(pretrained_model_name_or_path, config_name)
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, config_name)
        ):
            config_file = os.path.join(
                pretrained_model_name_or_path, subfolder, config_name
            )
        else:
            raise EnvironmentError(
                f"Error no file named {config_name} found in directory {pretrained_model_name_or_path}."
            )
    else:
        try:
            # Load from URL or cache if already cached
            config_file = hf_hub_download(
                pretrained_model_name_or_path,
                filename=config_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=use_auth_token,
                user_agent=user_agent,
                subfolder=subfolder,
                revision=revision,
            )

        except RepositoryNotFoundError as err:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier"
                " listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a"
                " token having permission to this repo with `use_auth_token` or log in with `huggingface-cli"
                " login`."
            ) from err
        except RevisionNotFoundError as err:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for"
                " this model name. Check the model page at"
                f" 'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
            ) from err
        except EntryNotFoundError as err:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {config_name}."
            ) from err
        except HTTPError as err:
            raise EnvironmentError(
                "There was a specific connection error when trying to load"
                f" {pretrained_model_name_or_path}:\n{err}"
            ) from err
        except ValueError as err:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                f" directory containing a {config_name} file.\nCheckout your internet connection or see how to"
                " run the library in offline mode at"
                " 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
            ) from err
        except EnvironmentError as err:
            raise EnvironmentError(
                f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a {config_name} file"
            ) from err

    try:
        # Load config dict
        assert config_file is not None
        config_dict = dict_from_json_file(config_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise EnvironmentError(
            f"It looks like the config file at '{config_file}' is not a valid JSON file."
        ) from err

    if return_unused_kwargs:
        return config_dict, kwargs

    return config_dict


def download_model(
    pretrained_model_name: str,
    auth_token: Optional[str] = os.environ["HUGGINGFACE_TOKEN"],
    cache_dir: Path = Path(DIFFUSERS_CACHE),
    resume_download: bool = True,
    revision: Optional[str] = None,
    local_files_only: bool = HF_HUB_OFFLINE,
    force_download: bool = False,
):
    "Download a model from the Hugging Face Hub"

    if not os.path.isdir(pretrained_model_name):
        config_dict = load_config(
            pretrained_model_name_or_path=pretrained_model_name,
            cache_dir=cache_dir,
            resume_download=resume_download,
            force_download=force_download,
            local_files_only=local_files_only,
            use_auth_token=auth_token,
            revision=revision,
        )
        # make sure we only download sub-folders and `diffusers` filenames
        folder_names = [k for k in config_dict.keys() if not k.startswith("_")]  # type: ignore
        allow_patterns = [os.path.join(k, "*") for k in folder_names]
        allow_patterns += [
            WEIGHTS_NAME,
            SCHEDULER_CONFIG_NAME,
            CONFIG_NAME,
            ONNX_WEIGHTS_NAME,
            config_name,
        ]

        # # make sure we don't download flax weights
        ignore_patterns = ["*.msgpack"]

        if is_safetensors_available() and not local_files_only:
            info = model_info(
                repo_id=pretrained_model_name,
                token=auth_token,
                revision=revision,
            )
            if is_safetensors_compatible(info):
                ignore_patterns.append("*.bin")
            else:
                # as a safety mechanism we also don't download safetensors if
                # not all safetensors files are there
                ignore_patterns.append("*.safetensors")
        else:
            ignore_patterns.append("*.safetensors")

        snapshot_download(
            repo_id=pretrained_model_name,
            cache_dir=cache_dir,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=auth_token,
            revision=revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )


def is_safetensors_compatible(info: ModelInfo) -> bool:
    "Check if the model is compatible with safetensors"

    filenames = set(sibling.rfilename for sibling in info.siblings)
    pt_filenames = set(filename for filename in filenames if filename.endswith(".bin"))
    safetensors_compatible = any(file.endswith(".safetensors") for file in filenames)
    for pt_filename in pt_filenames:
        prefix, raw = os.path.split(pt_filename)
        if raw == "pytorch_model.bin":
            # transformers specific
            sf_filename = os.path.join(prefix, "model.safetensors")
        else:
            sf_filename = pt_filename[: -len(".bin")] + ".safetensors"
        if safetensors_compatible and sf_filename not in filenames:
            logger.warning(f"{sf_filename} not found")
            safetensors_compatible = False
    return safetensors_compatible


def dict_from_json_file(json_file: Union[str, os.PathLike]):
    "Read a json file into a python dict."

    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)
