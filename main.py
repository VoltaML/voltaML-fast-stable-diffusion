import logging
import os
import shlex
import subprocess
import sys
import threading
import warnings
from argparse import ArgumentParser
from pathlib import Path

from core.install_requirements import (  # pylint: disable=wrong-import-position
    commit_hash,
    create_environment,
    in_virtualenv,
    install_pytorch,
    is_installed,
    version_check,
)

# Handle arguments passed to the script
app_args = [] if os.getenv("TESTING") == "1" else sys.argv[1:]
extra_args = os.getenv("EXTRA_ARGS")

if extra_args:
    app_args.extend(shlex.split(extra_args))

# Parse arguments
parser = ArgumentParser(
    prog="VoltaML Fast Stable Diffusion",
    epilog="""
VoltaML Fast Stable Diffusion - Accelerated Stable Diffusion inference
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
""",
)
parser.add_argument(
    "--log-level",
    default="INFO",
    help="Log level",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)
parser.add_argument("--ngrok", action="store_true", help="Use ngrok to expose the API")
parser.add_argument("--host", action="store_true", help="Expose the API to the network")
parser.add_argument("--in-container", action="store_true", help="Skip virtualenv check")
parser.add_argument(
    "--bot", action="store_true", help="Run in tandem with the Discord bot"
)
parser.add_argument(
    "--enable-r2",
    action="store_true",
    help="Enable Cloudflare R2 bucket upload support",
)

parser.add_argument(
    "--install-only",
    action="store_true",
    help="Only install requirements and exit",
)
args = parser.parse_args(args=app_args)

logging.basicConfig(level=args.log_level)
logger = logging.getLogger(__name__)

# Suppress some annoying logs
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.INFO)

# Create necessary folders
Path("data/aitemplate").mkdir(exist_ok=True, parents=True)
Path("data/onnx").mkdir(exist_ok=True)
Path("data/models").mkdir(exist_ok=True)
Path("data/outputs").mkdir(exist_ok=True)
Path("data/lora").mkdir(exist_ok=True)
Path("data/tensorrt").mkdir(exist_ok=True)

# Suppress some annoying warnings
warnings.filterwarnings("ignore", category=UserWarning)


def is_root():
    "Check if user has elevated privileges"

    try:
        is_admin = os.getuid() == 0  # type: ignore
    except AttributeError:
        import ctypes

        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore

    return is_admin


def main(exit_after_init: bool = False):
    "Run the API"

    # Attach ngrok if requested
    if args.ngrok:
        import nest_asyncio
        from pyngrok import ngrok

        ngrok_tunnel = ngrok.connect(5003)
        logger.info(f"Public URL: {ngrok_tunnel.public_url}")
        nest_asyncio.apply()

    # Start the bot if requested
    if args.bot:

        def bot_call():
            from bot.bot import ModularBot

            bot = ModularBot()
            bot.run(os.environ["DISCORD_BOT_TOKEN"])

        bot_thread = threading.Thread(target=bot_call)
        bot_thread.daemon = True
        bot_thread.start()

    # Start the API
    from uvicorn import run as uvicorn_run

    from api.app import app as api_app

    host = "0.0.0.0" if args.host else "127.0.0.1"

    if not exit_after_init:
        uvicorn_run(api_app, host=host, port=5003)
    else:
        logger.warning("Exit after initialization requested, exiting now")


def checks():
    "Check if the script is run from a virtual environment, if yes, check requirements"

    if not (is_root() or args.in_container):
        if not in_virtualenv():
            create_environment()

            logger.error("Please run the script from a virtual environment")
            sys.exit(1)

    # Install more user friendly logging
    if not is_installed("coloredlogs"):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "coloredlogs",
            ]
        )

    if not is_installed("packaging"):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "packaging",
            ]
        )

    if not is_installed("dotenv"):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "python-dotenv",
            ]
        )

    import dotenv

    dotenv.load_dotenv()

    # Check tokens
    if not os.getenv("HUGGINGFACE_TOKEN") and not args.install_only:
        logger.error(
            "No token provided. Please provide a token with HUGGINGFACE_TOKEN environment variable"
        )
        sys.exit(1)

    if args.bot and not args.install_only:
        if not os.getenv("DISCORD_BOT_TOKEN"):
            logger.error(
                "Bot start requested, but no Discord token provided. Please provide a token with DISCORD_BOT_TOKEN environment variable"
            )
            sys.exit(1)

    # Inject coloredlogs
    import coloredlogs

    coloredlogs.DEFAULT_LEVEL_STYLES = {
        **coloredlogs.DEFAULT_LEVEL_STYLES,
        "info": {"color": "magenta", "bright": True},
        "error": {"color": "red", "bright": True, "bold": True},
        "warning": {"color": "yellow", "bright": True, "bold": True},
    }

    coloredlogs.install(
        level=args.log_level,
        fmt="%(asctime)s | %(name)s | %(levelname)s » %(message)s",
        datefmt="%H:%M:%S",
    )

    # Check if we are up to date with the latest release
    version_check(commit_hash())

    # Install pytorch and api requirements
    install_pytorch()

    # Save the token to config
    from core import shared

    if not args.install_only:
        shared.hf_token = os.environ["HUGGINGFACE_TOKEN"]

    # Create the diffusers cache folder
    from diffusers.utils import DIFFUSERS_CACHE

    Path(DIFFUSERS_CACHE).mkdir(exist_ok=True, parents=True)

    from core.config import config

    logger.info(f"Device: {config.api.device}")
    logger.info(f"Precision: {'FP32' if config.api.use_fp32 else 'FP16'}")

    # Initialize R2 bucket if needed
    if args.enable_r2:
        from core import shared_dependent
        from core.extra.cloudflare_r2 import R2Bucket

        endpoint = os.environ["R2_ENDPOINT"]
        bucket_name = os.environ["R2_BUCKET_NAME"]

        shared_dependent.r2 = R2Bucket(endpoint=endpoint, bucket_name=bucket_name)


if __name__ == "__main__":
    print("Starting the API...")

    checks()

    try:
        main(exit_after_init=args.install_only)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, exiting...")
        sys.exit(0)
