import os

from bot.bot import ModularBot

bot = ModularBot()

bot.run(os.environ["DISCORD_BOT_TOKEN"])
