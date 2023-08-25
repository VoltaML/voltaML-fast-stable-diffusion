from typing import TYPE_CHECKING, Any, Dict, List

import discord
from aiohttp import ClientSession
from discord.ext import commands
from discord.ext.commands import Cog, Context

from bot import shared
from bot.helper import get_available_models, get_loaded_models
from core.types import InferenceBackend

if TYPE_CHECKING:
    from bot.bot import ModularBot


class Models(Cog):
    "Commands for interacting with the models"

    def __init__(self, bot: "ModularBot") -> None:
        super().__init__()
        self.bot = bot

    @commands.hybrid_command(name="loaded")
    @commands.has_permissions(administrator=True)
    async def loaded_models(self, ctx: Context) -> None:
        "Show models loaded in the API"

        async with ClientSession() as session:
            async with session.get("http://localhost:5003/api/models/loaded") as r:
                status = r.status
                response: List[Dict[str, Any]] = await r.json()

        models = []
        if status == 200:
            for model in response:
                models.append(f"{model['name']} - {model['backend']}")

            embed = discord.Embed(
                title="Loaded Models",
                description="\n".join(models)
                if len(models) > 0
                else "No models loaded",
            )
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"Error: {status}")

    @commands.hybrid_command(name="available", aliases=["refresh_models"])
    async def available_models(self, ctx: Context) -> None:
        "List all available models"

        available_models, status = await get_available_models()
        shared.models.set_cached_available_models(available_models)
        available_models, status = await shared.models.cached_available_models()

        loaded_models, status = await get_loaded_models()
        shared.models.set_cached_loaded_models(loaded_models)

        if status == 200:
            await ctx.send(
                "Available models:\n`{}`".format("\n".join(available_models))
            )
        else:
            await ctx.send(f"Error: {status}")

    @commands.hybrid_command(name="load")
    @commands.has_permissions(administrator=True)
    async def load_model(
        self,
        ctx: Context,
        model: str,
        backend: InferenceBackend = "PyTorch",
    ) -> None:
        "Load a model"

        message = await ctx.send(f"Loading model {model}...")

        async with ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/models/load",
                params={"model": model, "backend": backend},
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await message.edit(content=f"{response['message']}: {model}")
        else:
            await message.edit(content=f"Error: **{response.get('detail')}**")

    @commands.hybrid_command(name="unload")
    @commands.has_permissions(administrator=True)
    async def unload_model(self, ctx: Context, model: str) -> None:
        "Unload a model"

        message = await ctx.send(f"Unloading model {model}...")

        async with ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/models/unload",
                params={"model": model},
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await message.edit(content=f"{response['message']}: {model}")
        else:
            await message.edit(content=f"Error: {status}")


async def setup(bot: "ModularBot") -> None:
    "Will be called by the bot"

    ext = Models(bot)
    await bot.add_cog(ext)
