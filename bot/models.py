from typing import TYPE_CHECKING, Dict, List

import discord
from aiohttp import ClientSession
from discord.ext import commands
from discord.ext.commands import Cog, Context

from core.types import InferenceBackend, SupportedModel

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
                response: Dict[str, List[str]] = await r.json()

        if status == 200:
            for device in response.keys():
                models = list(response[device])
                embed = discord.Embed(title=f"GPU-{device}")
                embed.add_field(
                    name="Models",
                    value="\n".join(models),
                )
                await ctx.send(embed=embed)
        else:
            await ctx.send(f"Error: {status}")

    @commands.hybrid_command(name="available")
    @commands.has_permissions(administrator=True)
    async def available_models(self, ctx: Context) -> None:
        "List all available models"

        async with ClientSession() as session:
            async with session.get(
                "http://localhost:5003/api/models/available"
            ) as response:
                status = response.status
                data: List[Dict[str, str]] = await response.json()

        if status == 200:
            models = set([i["name"] for i in data])
            await ctx.send("Available models:\n{}".format("\n ".join(models)))
        else:
            await ctx.send(f"Error: {status}")

    @commands.hybrid_command(name="load")
    @commands.has_permissions(administrator=True)
    async def load_model(
        self,
        ctx: Context,
        model: SupportedModel,
        device: str = "cuda",
        backend: InferenceBackend = "PyTorch",
    ) -> None:
        "Load a model"

        message = await ctx.send(f"Loading model {model.value}...")

        async with ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/models/load",
                params={"model": model.value, "backend": backend, "device": device},
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await message.edit(content=f"{response['message']}: {model.value}")
        else:
            await message.edit(content=f"Error: **{response.get('detail')}**")

    @commands.hybrid_command(name="load-unsupported")
    @commands.has_permissions(administrator=True)
    async def load_model_unsupported(
        self,
        ctx: Context,
        model: str,
        device: str = "cuda",
        backend: InferenceBackend = "PyTorch",
    ) -> None:
        "Load a model"

        message = await ctx.send(f"Loading model {model}...")

        async with ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/models/load",
                params={"model": model, "backend": backend, "device": device},
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await message.edit(content=f"{response['message']}: {model}")
        else:
            await message.edit(content=f"Error: **{response.get('detail')}**")

    @commands.hybrid_command(name="unload")
    @commands.has_permissions(administrator=True)
    async def unload_model(
        self, ctx: Context, model: SupportedModel, gpu_id: int
    ) -> None:
        "Unload a model"

        message = await ctx.send(f"Unloading model {model.value}...")

        async with ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/models/unload",
                params={"model": model.value, "gpu_id": gpu_id},
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await message.edit(content=f"{response['message']}: {model.value}")
        else:
            await message.edit(content=f"Error: {status}")

    @commands.hybrid_command(name="unload-unspported")
    @commands.has_permissions(administrator=True)
    async def unload_model_unsupported(
        self, ctx: Context, model: str, gpu_id: int
    ) -> None:
        "Unload a model"

        message = await ctx.send(f"Unloading model {model}...")

        async with ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/models/unload",
                params={"model": model, "gpu_id": gpu_id},
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await message.edit(content=f"{response['message']}: {model}")
        else:
            await message.edit(content=f"Error: {status}")


async def setup(bot: "ModularBot") -> None:
    "Will be called by the bot"

    await bot.add_cog(Models(bot))
