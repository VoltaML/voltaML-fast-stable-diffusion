from typing import TYPE_CHECKING

import discord
from discord.ext import commands
from discord.ext.commands import Cog

if TYPE_CHECKING:
    from bot.bot import ModularBot


class Listeners(Cog):
    "Middlewares for the bot"

    def __init__(self, bot: "ModularBot") -> None:
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        "When the bot is connected and ready to operate"

        await self.bot.change_presence(
            status=discord.Status.online, activity=discord.Game("VoltaML - /dream")
        )


async def setup(bot: "ModularBot") -> None:
    "Load the cog"

    await bot.add_cog(Listeners(bot))
