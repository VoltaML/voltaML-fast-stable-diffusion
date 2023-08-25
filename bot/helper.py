import asyncio
import difflib
from typing import Any, Dict, List, Literal

import aiohttp
from aiohttp import ClientSession

from bot import shared as shared_bot
from core import shared


async def find_closest_model(model: str):
    """Find the closest model to the one provided"""
    models, _ = await shared_bot.models.cached_loaded_models()
    return difflib.get_close_matches(model, models, n=1, cutoff=0.1)[0]


async def inference_call(payload: Dict, target: Literal["txt2img"] = "txt2img"):
    "Call to the backend to generate an image"

    async def call():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://localhost:{shared.api_port}/api/generate/{target}",
                json=payload,
            ) as response:
                status = response.status
                response = await response.json()

        return status, response

    try:
        status, response = await call()
    except aiohttp.ClientOSError:
        await asyncio.sleep(0.5)
        status, response = await call()

    return status, response


async def get_available_models():
    "List all available models"

    async with ClientSession() as session:
        async with session.get(
            f"http://localhost:{shared.api_port}/api/models/available"
        ) as response:
            status = response.status
            data: List[Dict[str, Any]] = await response.json()
            models = [
                i["name"]
                for i in filter(
                    lambda model: (
                        model["valid"] is True
                        and (
                            model["backend"] == "PyTorch"
                            or model["backend"] == "AITemplate"
                        )
                    ),
                    data,
                )
            ]

    return models, status


async def get_loaded_models():
    "List all available models"

    async with ClientSession() as session:
        async with session.get(
            f"http://localhost:{shared.api_port}/api/models/loaded"
        ) as response:
            status = response.status
            data: List[Dict[str, Any]] = await response.json()
            models = [
                i["name"]
                for i in filter(
                    lambda model: (
                        model["valid"] is True
                        and (
                            model["backend"] == "PyTorch"
                            or model["backend"] == "AITemplate"
                        )
                    ),
                    data,
                )
            ]

    return models, status
