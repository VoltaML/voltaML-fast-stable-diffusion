from typing import TYPE_CHECKING, Dict, Literal

import discord
from aiohttp import ClientSession
from discord.ext import commands
from discord.ext.commands import Cog, Context

from core.types import SupportedModel

if TYPE_CHECKING:
    from bot.bot import ModularBot


class Models(Cog):
    "Commands for interacting with the models"

    def __init__(self, bot: "ModularBot") -> None:
        super().__init__()
        self.bot = bot

    @commands.hybrid_command(name="loaded")
    @commands.is_owner()
    async def loaded_models(self, ctx: Context) -> None:
        "Show models loaded in the API"

        async with ClientSession() as session:
            async with session.get("http://localhost:5003/api/models/loaded") as r:
                status = r.status
                response: Dict[str, Dict] = await r.json()

        if status == 200:
            devices = []
            for model in response:
                if response[model]["device"] not in devices:
                    devices.append(response[model]["device"])

            for device in devices:
                embed = discord.Embed(title=f"Loaded models on {device}")
                embed.add_field(
                    name="Models",
                    value="\n".join(
                        [
                            model
                            for model in response
                            if response[model]["device"] == device
                        ]
                    ),
                )

                await ctx.send(
                    embed=embed,
                )
        else:
            await ctx.send(f"Error: {status}")

    @commands.hybrid_command(name="avaliable")
    @commands.is_owner()
    async def avaliable_models(self, ctx: Context) -> None:
        "List all avaliable models"

        async with ClientSession() as session:
            async with session.get(
                "http://localhost:5003/api/models/avaliable"
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await ctx.send(f"Avaliable models: {', '.join(response)}")
        else:
            await ctx.send(f"Error: {status}")

    @commands.hybrid_command(name="load")
    @commands.is_owner()
    async def load_model(
        self,
        ctx: Context,
        model: SupportedModel,
        device: str = "cuda",
        backend: Literal["PyTorch", "TensorRT"] = "PyTorch",
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
            await message.edit(content=f"Model loaded: {response['message']}")
        else:
            await message.edit(content=f"Error: **{response.get('detail')}**")

    @commands.hybrid_command(name="unload")
    @commands.is_owner()
    async def unload_model(self, ctx: Context, model: SupportedModel) -> None:
        "Unload a model"

        message = await ctx.send(f"Unloading model {model.value}...")

        async with ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/models/unload",
                params={"model": model.value},
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await message.edit(content=f"Model unloaded: {model.value}")
        else:
            await message.edit(content=f"Error: {status}")


async def setup(bot: "ModularBot") -> None:
    "Will be called by the bot"

    await bot.add_cog(Models(bot))
