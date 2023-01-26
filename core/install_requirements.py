import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

skip_requirements = ["cuda-python"]
logger = logging.getLogger(__name__)


def check_requirements(path_to_requirements: str = "requirements.txt"):
    "Check if all requirements are installed"

    with open(path_to_requirements, encoding="utf-8", mode="r") as f:
        requirements = [i.split("==")[0] for i in f.read().splitlines()]

        for requirement in requirements:
            fixed_name = requirement.replace("-", "_").lower()
            if not is_installed(fixed_name):
                return False

        return True


def install_requirements(path_to_requirements: str = "requirements.txt"):
    "Check if requirements are installed, if not, install them"

    with open(path_to_requirements, encoding="utf-8", mode="r") as f:
        requirements = [i.split("==")[0] for i in f.read().splitlines()]

        try:
            for requirement in requirements:
                if (
                    requirement.startswith("#")
                    or requirement.startswith("--")
                    or requirement in skip_requirements
                ):
                    continue

                fixed_name = requirement.replace("-", "_").lower()
                if not is_installed(fixed_name):
                    logger.debug(f"Requirement {requirement} is not installed")
                    raise ImportError

        except ImportError:
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        path_to_requirements,
                    ]
                )
            except subprocess.CalledProcessError:
                logger.error(f"Failed to install requirements: {path_to_requirements}")
                sys.exit(1)


def install_pytorch():
    "Install necessary requirements for inference"

    install_requirements("requirements/pytorch.txt")
    install_requirements("requirements/api.txt")


def install_bot():
    "Install necessary requirements for the discord bot"

    install_requirements("requirements/bot.txt")


def install_tensorrt():
    "Install necessary requirements for TensorRT inference"

    install_requirements("requirements/tensorrt.txt")


def is_installed(package):
    "Check if a package is installed"

    try:
        spec = importlib.util.find_spec(package)
        if spec is None:
            logger.debug(f"Package {package} - {'not found'}")
    except ModuleNotFoundError:
        return False

    return spec is not None


def get_base_prefix_compat():
    "Get base/real prefix, or sys.prefix if there is none"
    return (
        getattr(sys, "base_prefix", None)
        or getattr(sys, "real_prefix", None)
        or sys.prefix
    )


def commit_hash():
    """
    Get the commit hash of the current repository

    Some parts taken from A111 repo (https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/645f4e7ef8c9d59deea7091a22373b2da2b780f2/launch.py#L20)
    """

    try:
        result = subprocess.run(
            "git rev-parse HEAD",
            shell=True,
            check=True,
            capture_output=True,
            stdout=subprocess.PIPE,
        )
        stored_commit_hash = result.stdout.decode(encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def is_up_to_date():
    "Check if the virtual environment is up to date"


def in_virtualenv():
    "Check if we are in a virtual environment"

    return get_base_prefix_compat() != sys.prefix


def create_environment():
    "Create a virtual environment"

    if virtualenv_exists():
        command = (
            "source venv/bin/activate"
            if sys.platform == "linux"
            else "venv\\Scripts\\activate.bat OR venv\\Scripts\\Activate.ps1"
        )
        logger.info(
            f"Virtual environment already exists, you just need to activate it - {command}"
        )
        sys.exit(1)

    if not in_virtualenv():
        logger.info("Creating virtual environment")
        python_executable = sys.executable
        try:
            subprocess.run(
                f"{python_executable} -m virtualenv venv",
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.error("Failed to create virtual environment")
            sys.exit(1)
    else:
        logger.info("Already in virtual environment")


def virtualenv_exists():
    "Check if the virtual environment exists"

    return Path("venv").exists()
