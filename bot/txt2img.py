import logging
import random
from typing import TYPE_CHECKING, Literal, Optional
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
    "Commands for generating images from text"

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
        aspect_ratio: Literal["16:9", "9:16", "1:1", "civitai"] = "1:1",
        seed: Optional[int] = None,
        backend: Literal["PyTorch", "TensorRT"] = "PyTorch",
        scheduler: Scheduler = Scheduler.default,
        use_default_negative_prompt: bool = True,
    ):
        "Generate an image from prompt"

        default_negative_prompt = "nswf, lowres, bad anatomy, ((bad hands)), text, error, ((missing fingers)), cropped, jpeg artifacts, worst quality, low quality, signature, watermark, blurry, deformed, extra ears, disfigured, mutation, censored, fused legs, bad legs, bad hands, missing fingers, extra digit, fewer digits, normal quality, username, artist name"

        if seed is None:
            seed = random.randint(0, 1000000)

        if aspect_ratio == "16:9":
            width = 680
            height = 384
        elif aspect_ratio == "9:16":
            width = 384
            height = 680
        elif aspect_ratio == "civitai":
            width = 512
            height = 704
        else:
            width = 512
            height = 512

        payload = {
            "data": {
                "prompt": prompt,
                "id": uuid4().hex,
                "negative_prompt": negative_prompt
                if not use_default_negative_prompt
                else negative_prompt + default_negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "batch_size": 1,
                "batch_count": 1,
            },
            "model": model.value,
            "scheduler": scheduler.value,
            "backend": backend,
        }

        message = await ctx.send("Dreaming...")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/txt2img/generate", json=payload
            ) as response:
                status = response.status
                response = await response.json()

        if status == 200:
            await message.edit(
                content=f"{ctx.author.mention} Done! Seed: {seed}, Time {response.get('time'):.2f}s"
            )
            await message.add_files(
                File(
                    convert_base64_to_bytes(response["images"][0]),
                    filename="dream.png",
                )
            )
        else:
            await message.edit(content=f"{ctx.author.mention} Dream failed - {status}")

        logging.info(f"Finished task {prompt} for {str(ctx.author)}")


async def setup(bot: "ModularBot"):
    "Will be called by the bot"

    await bot.add_cog(Inference(bot))
