import importlib.metadata
import importlib.util
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

renamed_requirements = {
    "opencv-contrib-python-headless": "cv2",
    "fastapi-analytics": "api_analytics",
    "cuda-python": "cuda",
    "open_clip_torch": "open_clip",
}
logger = logging.getLogger(__name__)


def check_requirements(path_to_requirements: str = "requirements.txt"):
    "Check if all requirements are installed"

    with open(path_to_requirements, encoding="utf-8", mode="r") as f:
        requirements = {}
        for i in f.read().splitlines():
            if "==" in i:
                requirements[i.split("==")[0]] = i.replace(i.split("==")[0], "").strip()
            elif ">=" in i:
                requirements[i.split(">=")[0]] = i.replace(i.split(">=")[0], "").strip()
            elif "<=" in i:
                requirements[i.split("<=")[0]] = i.replace(i.split("<=")[0], "").strip()

        for requirement in requirements:
            fixed_name = requirement.replace("-", "_").lower()
            if not is_installed(fixed_name, requirements[requirement]):
                return False

        return True


def install_requirements(path_to_requirements: str = "requirements.txt"):
    "Check if requirements are installed, if not, install them"

    with open(path_to_requirements, encoding="utf-8", mode="r") as f:
        requirements = {}
        for i in [r.strip() for r in f.read().splitlines()]:
            if "git+http" in i:
                logger.debug(f"Skipping git requirement (cannot check version): {i}")
                continue

            if "==" in i:
                requirements[i.split("==")[0]] = i.replace(i.split("==")[0], "").strip()
            elif ">=" in i:
                requirements[i.split(">=")[0]] = i.replace(i.split(">=")[0], "").strip()
            elif "<=" in i:
                requirements[i.split("<=")[0]] = i.replace(i.split("<=")[0], "").strip()
            else:
                requirements[i] = None

        try:
            for requirement in requirements:
                if requirement.startswith("#") or requirement.startswith("--"):
                    continue

                if requirement in renamed_requirements:
                    logger.debug(
                        f"Requirement {requirement} is renamed to {renamed_requirements[requirement]}"
                    )
                    requirement_name = renamed_requirements[requirement]
                else:
                    requirement_name = requirement

                logger.debug(f"Checking requirement: {requirement}")
                fixed_name = requirement_name.replace("-", "_").lower()
                if not is_installed(fixed_name, requirements[requirement]):
                    logger.debug(f"Requirement {requirement_name} is not installed")
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

    # Install pytorch
    if platform.system() == "Windows":
        if not is_installed("torch", version="==2.0.0+cu118") or not is_installed(
            "torchvision"
        ):
            logger.info("Installing PyTorch")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch==2.0.0+cu118",
                    "torchvision",
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu118",
                ]
            )
    elif platform.system() == "Darwin":
        if not is_installed("torch", version="==2.0.0") or not is_installed(
            "torchvision"
        ):
            logger.info("Installing PyTorch")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchvision"]
            )
    else:
        if (
            subprocess.run(  # pylint: disable=subprocess-run-check
                "rocminfo",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True,
            ).returncode
            == 0
        ):
            logger.info("ROCmInfo success, assuming user has AMD GPU")

            if not is_installed("torch", "==2.0.0+cu118") or not is_installed(
                "torchvision", "==0.15.1+cu118"
            ):
                logger.info("Installing PyTorch")

                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch==2.0.0",
                        "torchvision",
                        "--index-url",
                        "https://download.pytorch.org/whl/rocm5.4.2",
                    ]
                )

        else:
            logger.info("ROCmInfo check failed, assuming user has NVIDIA GPU")

            if not is_installed("torch", "==2.0.0+cu118") or not is_installed(
                "torchvision", "==0.15.1+cu118"
            ):
                logger.info("Installing PyTorch")

                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch==2.0.0",
                        "torchvision",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu118",
                    ]
                )

    # Install other requirements
    install_requirements("requirements/pytorch.txt")
    install_requirements("requirements/api.txt")
    install_requirements("requirements/interrogation.txt")


def install_bot():
    "Install necessary requirements for the discord bot"

    install_requirements("requirements/bot.txt")


def install_tensorrt():
    "Install necessary requirements for TensorRT inference"

    install_requirements("requirements/tensorrt.txt")


def is_installed(package: str, version: Optional[str] = None):
    "Check if a package is installed"

    try:
        spec = importlib.util.find_spec(package)
        if spec is None:
            raise ModuleNotFoundError

        if version is not None:
            from packaging import version as packaging_version

            version_number = version.split("=")[-1]
            version_type = version[:2]
            required_version = packaging_version.parse(version_number)
            current_version = packaging_version.parse(
                importlib.metadata.version(package)
            )

            logger.debug(
                f"Required version: {required_version} - Current version: {current_version} - version type: {version_type}"
            )

            if version_type == "==":
                assert current_version == required_version
            elif version_type == ">=":
                assert current_version >= required_version
            elif version_type == "<=":
                assert current_version <= required_version

    except AssertionError:
        logger.debug(
            f"Package {package} - found - incorrect version - {version} - {importlib.metadata.version(package)}"
        )
        return False

    except ModuleNotFoundError:
        logger.debug(f"Package {package} - not found")
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
        )
        stored_commit_hash = result.stdout.decode(encoding="utf-8").strip()
    except subprocess.CalledProcessError:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def version_check(commit: str):
    """
    Check if the local version is up to date

    Taken from: https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/645f4e7ef8c9d59deea7091a22373b2da2b780f2/launch.py#L134
    """

    try:
        import requests

        current_branch = (
            subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True)
            .decode("utf-8")
            .strip()
        )

        origin = (
            subprocess.check_output("git config --get remote.origin.url", shell=True)
            .decode("utf-8")
            .strip()
        )
        username = origin.split("/")[-2]
        project = origin.split("/")[-1].split(".")[  # pylint: disable=use-maxsplit-arg
            0
        ]

        commits = requests.get(
            f"https://api.github.com/repos/{username}/{project}/branches/{current_branch}",
            timeout=5,
        ).json()
        print(f"Current commit: {commit}")
        if commit not in ("<none>", commits["commit"]["sha"]):
            print("--------------------------------------------------------")
            print("| You are not up to date with the most recent release. |")
            print("| Consider running `git pull` to update.               |")
            print("--------------------------------------------------------")
        elif commits["commit"]["sha"] == commit:
            print("You are up to date with the most recent release.")
        else:
            print("Not a git clone, can't perform version check.")
    except Exception as e:  # pylint: disable=broad-except
        logger.debug(f"Version check failed: {e}")
        logger.info(
            "No git repo found, assuming that we are in containerized environment"
        )


def is_up_to_date():
    "Check if the virtual environment is up to date"


def in_virtualenv():
    "Check if we are in a virtual environment"

    return get_base_prefix_compat() != sys.prefix


def create_environment():
    "Create a virtual environment"

    command = (
        "source venv/bin/activate"
        if sys.platform == "linux"
        else "venv\\Scripts\\activate.bat OR venv\\Scripts\\Activate.ps1"
    )

    if virtualenv_exists():
        logger.info(
            f"Virtual environment already exists, you just need to activate it with '{command}', then run the script again"
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
            logger.info(
                f"Virtual environment created, please activate it with '{command}', then run the script again"
            )
        except subprocess.CalledProcessError:
            logger.error("Failed to create virtual environment")
            sys.exit(1)
    else:
        logger.info("Already in virtual environment")


def virtualenv_exists():
    "Check if the virtual environment exists"

    return Path("venv").exists()
