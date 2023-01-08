import logging

from coloredlogs import install as coloredlogs_install
from uvicorn import run as uvicorn_run

from api.app import app as api_app

coloredlogs_install(level="INFO")
root_logger = logging.getLogger()


def main():
    uvicorn_run(api_app, host="0.0.0.0", port=5003)


if __name__ == "__main__":
    main()
