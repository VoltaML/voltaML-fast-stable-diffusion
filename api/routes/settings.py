import logging

from fastapi import APIRouter

from core import config

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post("/save")
async def save_configuration(settings: config.Configuration):
    "Receive settings from the frontend and save them to the config file"

    config.config = settings
    config.save_config(config.config)

    return {"message": "success"}


@router.get("/")
async def get_configuration():
    "Return the current configuration to the frontend"

    logger.debug(f"Sending configuration to frontend: {config.config}")
    return config.config
