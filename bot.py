import os

from coloredlogs import install as install_coloredlogs

from bot.bot import ModularBot

bot = ModularBot()

install_coloredlogs("INFO")

bot.run(os.environ["DISCORD_BOT_TOKEN"])
