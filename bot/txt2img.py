import asyncio
import logging
import random
from typing import TYPE_CHECKING, Dict, Literal, Optional
from uuid import uuid4

import aiohttp
import discord
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from discord import File
from discord.ext import commands
from discord.ext.commands import Cog, Context

from core.config import config
from core.types import SupportedModel
from core.utils import convert_base64_to_bytes

if TYPE_CHECKING:
    from bot.bot import ModularBot


async def dream_call(payload: Dict):
    "Call to the backend to generate an image"

    async def call():
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:5003/api/generate/txt2img", json=payload
            ) as response:
                status = response.status
                response = await response.json()

        return status, response

    try:
        status, response = await call()
    except aiohttp.ClientOSError:
        await asyncio.sleep(0.5)
        status, response = await call()

    return status, response


class Inference(Cog):
    "Commands for generating images from text"

    def __init__(self, bot: "ModularBot") -> None:
        self.bot = (bot,)
        self.queue_number: int = 0

    @commands.hybrid_command(name="reset-queue")
    @commands.is_owner()
    async def reset_queue(self, ctx: Context):
        "Reset the queue number"

        self.queue_number = 0
        await ctx.send("✅ Queue reset!")

    @commands.hybrid_command(name="dream-unsupported")
    async def dream_unsupported(
        self,
        ctx: Context,
        prompt: str,
        model: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.0,
        steps: Literal[25, 30, 50] = 30,
        aspect_ratio: Literal["16:9", "9:16", "1:1"] = "1:1",
        seed: Optional[int] = None,
        scheduler: KarrasDiffusionSchedulers = KarrasDiffusionSchedulers.EulerAncestralDiscreteScheduler,
        use_default_negative_prompt: bool = True,
        verbose: bool = False,
    ):
        "Generate an image from prompt"

        self.queue_number += 1
        default_negative_prompt = "nsfw, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless"

        if model == SupportedModel.SynthWave:
            prompt = "snthwve style, " + prompt
        elif model == SupportedModel.OpenJourney:
            prompt = "mdjrny-4, " + prompt
        elif model == SupportedModel.InkpunkDiffusion:
            prompt = "nvinkpunk, " + prompt

        if seed is None:
            seed = random.randint(0, 1000000)

        if aspect_ratio == "16:9":
            width = 680
            height = 384
        elif aspect_ratio == "9:16":
            width = 384
            height = 680
        # elif aspect_ratio == "civitai":
        #     width = 512
        #     height = 704
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
                "scheduler": scheduler.value,
            },
            "model": model,
            "save_image": False,
        }

        message = await ctx.send(
            "Dreaming... **Queue number: " + str(self.queue_number) + "**"
        )

        try:
            status, response = await dream_call(payload=payload)
        except Exception as e:
            self.queue_number -= 1
            raise e

        self.queue_number -= 1

        if status == 200:
            if verbose:
                embed = discord.Embed(
                    color=discord.Color.green(),
                )
                embed.add_field(name="Seed", value=seed)
                embed.add_field(name="Time", value=f"{response.get('time'):.2f}s")
                embed.add_field(name="Model", value=model)
                embed.add_field(
                    name="Negative Prompt",
                    value=negative_prompt
                    if not use_default_negative_prompt
                    else "*Default*"
                    + (" + " if negative_prompt else "")
                    + negative_prompt,
                )
                embed.add_field(name="Guidance Scale", value=guidance_scale)
                embed.add_field(name="Steps", value=steps)
                embed.add_field(name="Aspect Ratio", value=aspect_ratio)

                await message.edit(embed=embed)

            await message.edit(
                content=f"{ctx.author.mention} - **{prompt}**, Time: {response.get('time'):.2f}s, Seed: {seed}"
            )
            await message.add_files(
                File(
                    convert_base64_to_bytes(response["images"][0]),
                    filename=f"{seed}.png",
                )
            )
        else:
            if response.get("detail"):
                await message.edit(
                    content=f"{ctx.author.mention} Dream failed - **{response.get('detail')}**"
                )
            else:
                await message.edit(
                    content=f"{ctx.author.mention} Dream failed - {status}"
                )

        logging.info(f"Finished task {prompt} for {str(ctx.author)}")

    @commands.hybrid_command(name="dream")
    async def dream(
        self,
        ctx: Context,
        prompt: str,
        model: SupportedModel,
        negative_prompt: str = "",
        guidance_scale: float = 7.0,
        steps: Literal[25, 30, 50] = 30,
        resolution: Literal[
            "512x512",
            "1024x1024",
            "512x912",
            "912x512",
            "1920x1080",
            "1080x1920",
            "1280x720",
            "720x1280",
            "768x768",
        ] = "512x512",
        seed: Optional[int] = None,
        scheduler: KarrasDiffusionSchedulers = config.bot.default_scheduler,
        use_default_negative_prompt: bool = config.bot.use_default_negative_prompt,
        verbose: bool = config.bot.verbose,
    ):
        "Generate an image from prompt"

        self.queue_number += 1
        default_negative_prompt = "nsfw, (((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless"

        if model == SupportedModel.SynthWave:
            prompt = "snthwve style, " + prompt
        elif model == SupportedModel.OpenJourney:
            prompt = "mdjrny-4, " + prompt
        elif model == SupportedModel.InkpunkDiffusion:
            prompt = "nvinkpunk, " + prompt

        if seed is None:
            seed = random.randint(0, 1000000)

        width, height = [int(i) for i in resolution.split("x")]

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
                "scheduler": scheduler.value,
            },
            "model": model.value,
            "save_image": False,
        }

        message = await ctx.send(
            "Dreaming... **Queue number: " + str(self.queue_number) + "**"
        )

        try:
            status, response = await dream_call(payload=payload)
        except Exception as e:
            self.queue_number -= 1
            raise e

        self.queue_number -= 1

        if status == 200:
            if verbose:
                embed = discord.Embed(
                    color=discord.Color.green(),
                )
                embed.add_field(name="Seed", value=seed)
                embed.add_field(name="Time", value=f"{response.get('time'):.2f}s")
                embed.add_field(name="Model", value=model.value)
                embed.add_field(
                    name="Negative Prompt",
                    value=negative_prompt
                    if not use_default_negative_prompt
                    else "*Default*"
                    + (" + " if negative_prompt else "")
                    + negative_prompt,
                )
                embed.add_field(name="Guidance Scale", value=guidance_scale)
                embed.add_field(name="Steps", value=steps)
                embed.add_field(name="Resolution", value=resolution)

                await message.edit(embed=embed)

            await message.edit(
                content=f"{ctx.author.mention} - **{prompt}**, Time: {response.get('time'):.2f}s, Seed: {seed}"
            )
            await message.add_files(
                File(
                    convert_base64_to_bytes(response["images"][0]),
                    filename=f"{seed}.png",
                )
            )
        else:
            if response.get("detail"):
                await message.edit(
                    content=f"{ctx.author.mention} Dream failed - **{response.get('detail')}**"
                )
            else:
                await message.edit(
                    content=f"{ctx.author.mention} Dream failed - {status}"
                )

        logging.info(f"Finished task {prompt} for {str(ctx.author)}")


async def setup(bot: "ModularBot"):
    "Will be called by the bot"

    await bot.add_cog(Inference(bot))
