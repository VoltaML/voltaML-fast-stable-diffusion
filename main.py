from argparse import ArgumentParser

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


def main():
    "Run the API"

    uvicorn_run(api_app, host="0.0.0.0", port=5003)


if __name__ == "__main__":
    main()
