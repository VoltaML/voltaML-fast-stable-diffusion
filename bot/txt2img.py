import logging
import random
import re
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

import discord
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from discord import File
from discord.ext import commands
from discord.ext.commands import Cog, Context

from bot.helper import find_closest_model, inference_call
from bot.shared_api import config
from core.utils import convert_base64_to_bytes

if TYPE_CHECKING:
    from bot.bot import ModularBot

logger = logging.getLogger(__name__)

pattern = r"data:image/[\w]+;base64,"


class Inference(Cog):
    "Commands for generating images from text"

    def __init__(self, bot: "ModularBot") -> None:
        self.bot = bot

    @commands.hybrid_command(name="text2image")
    async def dream_unsupported(
        self,
        ctx: Context,
        prompt: str,
        model: str,
        negative_prompt: str = "",
        guidance_scale: float = config.default_cfg,
        steps: int = config.default_steps,
        width: int = config.default_width,
        height: int = config.default_height,
        count: int = config.default_count,
        seed: Optional[int] = None,
        scheduler: KarrasDiffusionSchedulers = config.default_scheduler,
        verbose: bool = config.default_verbose,
    ):
        "Generate an image from prompt"

        if seed is None:
            seed = random.randint(0, 1000000)

        if config.max_width < width or config.max_height < height:
            return await ctx.send(
                f"Image size is too big, maximum size is {config.max_width}x{config.max_height}"
            )

        if config.max_count < count:
            return await ctx.send(
                f"Image count is too big, maximum count is {config.max_count}"
            )

        prompt = prompt + config.extra_prompt
        negative_prompt = negative_prompt + config.extra_negative_prompt

        try:
            model = await find_closest_model(model)
        except IndexError:
            await ctx.send(f"No loaded model that is close to `{model}` found")
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
                "batch_count": count,
                "scheduler": scheduler.value,
            },
            "model": model,
            "save_image": False,
        }

        message = await ctx.send(f"Generating image with `{model}`...")

        try:
            status, response = await inference_call(payload=payload)
        except Exception as e:
            raise e

        if status == 200:
            if verbose:
                embed = discord.Embed(
                    color=discord.Color.green(),
                )
                embed.add_field(name="Seed", value=seed)
                embed.add_field(name="Time", value=f"{response.get('time'):.2f}s")
                embed.add_field(name="Model", value=model)
                embed.add_field(name="Negative Prompt", value=negative_prompt)
                embed.add_field(name="Guidance Scale", value=guidance_scale)
                embed.add_field(name="Steps", value=steps)
                embed.add_field(name="Width", value=width)
                embed.add_field(name="Height", value=height)

                await message.edit(embed=embed)

            await message.edit(
                content=f"{ctx.author.mention} - **{prompt}**, Time: {response.get('time'):.2f}s, Seed: {seed}"
            )
            file_array = [
                File(
                    convert_base64_to_bytes(re.sub(pattern, "", x)),
                    filename=f"{seed}.png",
                )
                for x in response["images"]
            ]

            await message.add_files(*file_array[len(file_array) - count :])
        else:
            if response.get("detail"):
                await message.edit(
                    content=f"{ctx.author.mention} Dream failed - **{response.get('detail')}**"
                )
            else:
                await message.edit(
                    content=f"{ctx.author.mention} Dream failed - {status}"
                )

        logger.info(f"Finished task {prompt} for {str(ctx.author)}")


async def setup(bot: "ModularBot"):
    "Will be called by the bot"

    await bot.add_cog(Inference(bot))
