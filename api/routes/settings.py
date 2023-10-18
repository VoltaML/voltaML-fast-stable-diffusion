import logging
import os

from fastapi import APIRouter

from core import config
from core.config._config import update_config

router = APIRouter(tags=["settings"])

logger = logging.getLogger(__name__)


@router.post("/save")
async def save_configuration(settings: config.Configuration):
    "Receive settings from the frontend and save them to the config file"

    reload_required = False
    if config.config.api.device != settings.api.device:
        logger.info(f"Device was changed to {settings.api.device}")
        reload_required = True
    if config.config.api.data_type != settings.api.data_type:
        logger.info(f"Precision changed to {settings.api.data_type}")
        reload_required = True
    if config.config.api != settings.api:
        reload_required = True

    if reload_required:
        logger.info(
            "API settings changed, you might need to reload your models for these changes to take effect"
        )

    update_config(config.config, settings)
    config.save_config(config.config)

    logger.info("Config was updated and saved to disk")

    return {"message": "success"}


@router.get("/")
async def get_configuration():
    "Return the current configuration to the frontend"

    logger.debug(f"Sending configuration to frontend: {config.config}")
    return config.config


@router.post("/inject-var-into-dotenv")
async def set_hf_token(key: str, value: str):
    "Set the HuggingFace token in the environment variables and in the .env file"

    from core.functions import inject_var_into_dotenv

    inject_var_into_dotenv(key, value)
    return {"message": "success"}


@router.get(
    "/hf-whoami",
)
async def hf_whoami():
    "Return the current HuggingFace user"

    from huggingface_hub import HfApi

    api = HfApi()

    return api.whoami(token=os.getenv("HUGGINGFACE_TOKEN"))
