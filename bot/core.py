from typing import TYPE_CHECKING

from discord.ext import commands
from discord.ext.commands import Cog, Context

if TYPE_CHECKING:
    from bot.bot import ModularBot


class Core(Cog):
    "Core commands"

    def __init__(self, bot: "ModularBot") -> None:
        self.bot = bot

    @commands.hybrid_command(name="sync")
    @commands.has_permissions(administrator=True)
    async def sync(self, ctx: Context):
        "Sync slash commands with the API"

        await self.bot.sync()
        await ctx.send("✅ Synced slash commands!")


async def setup(bot: "ModularBot"):
    "Will be called by the bot"

    await bot.add_cog(Core(bot))
