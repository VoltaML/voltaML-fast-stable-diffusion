import logging
import os
import subprocess
import sys
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

parser = ArgumentParser()
parser.add_argument(
    "--log-level",
    default="INFO",
    help="Log level",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)
parser.add_argument("--ngrok", action="store_true", help="Use ngrok to expose the API")
parser.add_argument("--host", action="store_true", help="Expose the API to the network")
parser.add_argument(
    "--token",
    help="Token to use for the API",
    default=os.environ["HUGGINGFACE_TOKEN"],
    type=str,
)
parser.add_argument("--in-container", action="store_true", help="Skip virtualenv check")
parser.add_argument("--low-vram", action="store_true", help="Use low VRAM mode")
args = parser.parse_args()


# Apply the token
os.environ["HUGGINGFACE_TOKEN"] = args.token

logging.basicConfig(level=args.log_level)
logger = logging.getLogger(__name__)

if not args.token:
    logger.error(
        "No token provided. Please provide a token with --token or set the HUGGINGFACE_TOKEN environment variable"
    )
    sys.exit(1)

# Suppress some annoying logs
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)
logging.getLogger("xformers").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

# Create necessary folders
Path("data/aitemplate").mkdir(exist_ok=True, parents=True)
Path("data/models").mkdir(exist_ok=True)
Path("engine").mkdir(exist_ok=True)
Path("onnx").mkdir(exist_ok=True)


def is_root():
    "Check if user has elevated privileges"
    try:
        is_admin = os.getuid() == 0  # type: ignore
    except AttributeError:
        import ctypes

        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0  # type: ignore

    return is_admin


def main():
    "Run the API"

    if args.ngrok:
        import nest_asyncio
        from pyngrok import ngrok

        ngrok_tunnel = ngrok.connect(5003)
        logger.info(f"Public URL: {ngrok_tunnel.public_url}")
        nest_asyncio.apply()

    from uvicorn import run as uvicorn_run

    from api.app import app as api_app

    host = "0.0.0.0" if args.host else "127.0.0.1"
    uvicorn_run(api_app, host=host, port=5003)


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

    shared.hf_token = args.token

    # Create the diffusers cache folder
    from diffusers.utils import DIFFUSERS_CACHE

    Path(DIFFUSERS_CACHE).mkdir(exist_ok=True, parents=True)

    # Config
    from core.config import config

    config.low_vram = True if args.low_vram else bool(os.environ.get("LOW_VRAM", False))
    if config.low_vram:
        logger.warning("Using low VRAM mode")


if __name__ == "__main__":
    print("Starting the API...")

    checks()
    main()
