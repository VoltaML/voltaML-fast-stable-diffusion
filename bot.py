import os

from coloredlogs import install as install_coloredlogs

from bot.bot import ModularBot

bot = ModularBot()

if __name__ == "__main__":
    print(
        """
    VoltaML Bot - Discord Bot for Stable Diffusion inference
    Copyright (C) 2023-present Stax124

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    """
    )

    install_coloredlogs("INFO")

    bot.run(os.environ["DISCORD_BOT_TOKEN"])
