import logging

from fastapi import APIRouter

from core import config
from core.config.config import update_config

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/save")
async def save_configuration(settings: config.Configuration):
    "Receive settings from the frontend and save them to the config file"

    reload_required = False
    if config.config.api.device_id != settings.api.device_id:
        logger.info(f"Device ID was changed to {settings.api.device_id}")
        reload_required = True
    if config.config.api.device_type != settings.api.device_type:
        logger.info(f"Device type was changed to {settings.api.device_type}")
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
