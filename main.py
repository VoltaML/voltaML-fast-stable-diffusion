import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from core.install_requirements import (
    commit_hash,
    create_environment,
    in_virtualenv,
    install_pytorch,
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
args = parser.parse_args()

logging.basicConfig(level=args.log_level)
logger = logging.getLogger(__name__)

PILLogger = logging.getLogger("PIL.PngImagePlugin")
PILLogger.setLevel(logging.INFO)
xFormersLogger = logging.getLogger("xformers")
xFormersLogger.setLevel(logging.ERROR)

traced_unet_folder = Path("traced_unet")
traced_unet_folder.mkdir(exist_ok=True)


def main():
    "Run the API"
    import torch.backends.cudnn

    # Enable best cudnn functions
    torch.backends.cudnn.benchmark = True

    from coloredlogs import install as coloredlogs_install

    coloredlogs_install(level=args.log_level)

    if args.ngrok:
        import nest_asyncio
        from pyngrok import ngrok

        ngrok_tunnel = ngrok.connect(5003)
        logger.info(f"Public URL: {ngrok_tunnel.public_url}")
        nest_asyncio.apply()

    from uvicorn import run as uvicorn_run

    from api.app import app as api_app

    uvicorn_run(api_app, host="0.0.0.0", port=5003)


def checks():
    "Check if the script is run from a virtual environment, if yes, check requirements"

    if not in_virtualenv():
        create_environment()

        logger.error("Please run the script from a virtual environment")
        sys.exit(1)

    version_check(commit_hash())

    install_pytorch()


if __name__ == "__main__":
    checks()
    main()
