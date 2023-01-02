import io
from typing import TYPE_CHECKING

from discord import File
from discord.ext import commands
from discord.ext.commands import Cog, Context

from api.types import Txt2imgJob
from core.inference.pytorch import PyTorchInferenceModel

if TYPE_CHECKING:
    from bot.bot import ModularBot


class Inference(Cog):
    def __init__(self, bot: "ModularBot") -> None:
        self.bot = bot
        self.model = PyTorchInferenceModel("Linaqruf/anything-v3.0")
        if self.model.model:
            self.model.model.enable_xformers_memory_efficient_attention()

    @commands.hybrid_command(name="dream")
    async def dream(
        self,
        ctx: Context,
        prompt: str,
        negative_prompt: str = "",
        guidance_scale: float = 7.0,
    ):
        job = Txt2imgJob(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
        )
        message = await ctx.send("Dreaming...")

        images = self.model.generate(job)

        with io.BytesIO() as image_binary:
            images[0].save(image_binary, "PNG")
            image_binary.seek(0)
            await message.edit(content="Done!")
            await message.add_files(File(fp=image_binary, filename="dream.png"))


async def setup(bot: "ModularBot"):
    await bot.add_cog(Inference(bot))
