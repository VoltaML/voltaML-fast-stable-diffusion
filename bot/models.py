from typing import TYPE_CHECKING

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
    async def loaded_models(self, ctx: Context) -> None:
        "Show models loaded in the API"

        async with ClientSession() as session:
            async with session.get(
                "http://localhost:5003/api/models/loaded"
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await ctx.send(f"Loaded models: {', '.join(response)}")
        else:
            await ctx.send(f"Error: {status}")

    @commands.hybrid_command(name="avaliable")
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

    @commands.hybrid_command(name="unload")
    async def unload_model(self, ctx: Context, model: SupportedModel) -> None:
        "Unload a model"

        async with ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/models/unload",
                json={"model": model.value},
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await ctx.send(f"Model unloaded: {response['message']}")
        else:
            await ctx.send(f"Error: {status}")


async def setup(bot: "ModularBot") -> None:
    "Will be called by the bot"

    await bot.add_cog(Models(bot))
