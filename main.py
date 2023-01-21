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
args = parser.parse_args()

coloredlogs_install(level=args.log_level)

traced_unet_folder = Path("traced_unet")
traced_unet_folder.mkdir(exist_ok=True)


def main():
    "Run the API"

    import torch.backends.cudnn

    # Enable best cudnn functions
    torch.backends.cudnn.benchmark = True

    uvicorn_run(api_app, host="0.0.0.0", port=5003)


if __name__ == "__main__":
    main()
