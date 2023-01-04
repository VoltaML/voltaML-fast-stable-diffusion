import logging
import random
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

import aiohttp
from discord import File
from discord.ext import commands
from discord.ext.commands import Cog, Context

from core.types import Scheduler, SupportedModel
from core.utils import convert_base64_to_bytes

if TYPE_CHECKING:
    from bot.bot import ModularBot

class Inference(Cog):
    def __init__(self, bot: "ModularBot") -> None:
        self.bot = bot

    @commands.hybrid_command(name="dream")
    async def dream(
        self,
        ctx: Context,
        prompt: str,
        model: SupportedModel = SupportedModel.AnythingV3,
        negative_prompt: str = "",
        guidance_scale: float = 7.0,
        steps: int = 25,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ):
        if seed is None:
            seed = random.randint(0, 1000000)
        
        if width % 8 != 0 or height % 8 != 0:
            await ctx.send("Width and height must be divisible by 8")
            return

        payload = {
                    "data": {
                        "prompt": prompt,
                        "id": uuid4().hex,
                        "negative_prompt": negative_prompt,
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "seed": seed,
                        "batch_size": 1,
                        "batch_count": 1,
                    },
                    "model": model.value,
                    "scheduler": Scheduler.default.value,
                    "backend": "PyTorch",
                    }

        message = await ctx.send("Dreaming...")
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8080/api/txt2img/generate", json=payload) as response:
                status = response.status
                response = await response.json()
                
        
        if response.get("images"):
            await message.edit(content=f"{ctx.author.mention} Done! Seed: {seed}")
            await message.add_files(File(convert_base64_to_bytes(response["images"][0]), filename=f"dream.png"))
        else:
            await message.edit(content=f"{ctx.author.mention} Dream failed - {status}")

        logging.info(f"Finished task {prompt} for {ctx.author.__str__()}")


async def setup(bot: "ModularBot"):
    await bot.add_cog(Inference(bot))
