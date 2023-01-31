from typing import TYPE_CHECKING, Dict, List, Literal

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
    @commands.has_permissions(administrator=True)
    async def loaded_models(self, ctx: Context) -> None:
        "Show models loaded in the API"

        async with ClientSession() as session:
            async with session.get("http://localhost:5003/api/models/loaded") as r:
                status = r.status
                response: Dict[str, List] = await r.json()

        if status == 200:
            models = []
            for device in response.keys():
                for model in response[device]:
                    models.append(model)

            models = set(models)

            embed = discord.Embed(title=f"Loaded models: {len(models)}")
            embed.add_field(
                name="Models",
                value="\n".join(models),
            )

            await ctx.send(
                embed=embed,
            )
        else:
            await ctx.send(f"Error: {status}")

    @commands.hybrid_command(name="avaliable")
    @commands.has_permissions(administrator=True)
    async def avaliable_models(self, ctx: Context) -> None:
        "List all avaliable models"

        async with ClientSession() as session:
            async with session.get(
                "http://localhost:5003/api/models/avaliable"
            ) as response:
                status = response.status
                data: List[Dict[str, str]] = await response.json()

        if status == 200:
            models = set([i["name"] for i in data])
            await ctx.send("Avaliable models:\n{}".format("\n ".join(models)))
        else:
            await ctx.send(f"Error: {status}")

    @commands.hybrid_command(name="load")
    @commands.has_permissions(administrator=True)
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
    @commands.has_permissions(administrator=True)
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
