from uvicorn import run as uvicorn_run

from api.app import app as api_app


def main():
    uvicorn_run(api_app, host="0.0.0.0", port=5003)


if __name__ == "__main__":
    main()
