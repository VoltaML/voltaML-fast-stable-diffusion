import logging
from argparse import ArgumentParser
from pathlib import Path

from coloredlogs import install as coloredlogs_install
from uvicorn import run as uvicorn_run

from api.app import app as api_app

parser = ArgumentParser()
parser.add_argument(
    "--log-level",
    default="INFO",
    help="Log level",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)
parser.add_argument("--ngrok", action="store_true", help="Use ngrok to expose the API")
args = parser.parse_args()

coloredlogs_install(level=args.log_level)

logger = logging.getLogger(__name__)

traced_unet_folder = Path("traced_unet")
traced_unet_folder.mkdir(exist_ok=True)


def main():
    "Run the API"

    import torch.backends.cudnn

    # Enable best cudnn functions
    torch.backends.cudnn.benchmark = True

    if args.ngrok:
        import nest_asyncio
        from pyngrok import ngrok

        ngrok_tunnel = ngrok.connect(5003)
        logger.info(f"Public URL: {ngrok_tunnel.public_url}")
        nest_asyncio.apply()

    uvicorn_run(api_app, host="0.0.0.0", port=5003)


if __name__ == "__main__":
    main()
