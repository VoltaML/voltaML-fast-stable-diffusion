from typing import TYPE_CHECKING, Dict

import discord
from aiohttp import ClientSession
from discord.ext import commands
from discord.ext.commands import Cog, Context

if TYPE_CHECKING:
    from bot.bot import ModularBot


class Hardware(Cog):
    "Hardware commands"

    def __init__(self, bot: "ModularBot"):
        self.bot = bot

    @commands.hybrid_command(name="gpus")
    @commands.is_owner()
    async def gpus(self, ctx: Context):
        "List all available GPUs"

        async with ClientSession() as session:
            async with session.get("http://localhost:5003/api/hardware/gpus") as resp:
                status = resp.status
                data: Dict[str, Dict] = await resp.json()

        if status != 200:
            await ctx.send("Something went wrong")
            return

        embed = discord.Embed(title="GPUs", color=0x00FF00)
        for i, gpu in data.items():
            embed.add_field(
                name=f"GPU {i}",
                value=(
                    f"Name: {gpu['name']}\n" f"Total memory: {gpu['total_memory']}\n"
                ),
            )

        await ctx.send(embed=embed)

    @commands.hybrid_command(name="clean-memory")
    @commands.is_owner()
    async def clean(self, ctx: Context):
        "Free the memory manually"

        await ctx.defer()

        async with ClientSession() as session:
            async with session.post("http://localhost:5003/api/models/memory-cleanup") as resp:
                status = resp.status

        if status != 200:
            await ctx.send("Something went wrong")
            return

        await ctx.send("Cleaned up the memory")


async def setup(bot: "ModularBot"):
    "Will be loaded by bot"

    await bot.add_cog(Hardware(bot))
