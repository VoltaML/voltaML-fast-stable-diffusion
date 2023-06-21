import importlib.metadata
import importlib.util
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import List, Optional, Union

renamed_requirements = {
    "opencv-contrib-python-headless": "cv2",
    "fastapi-analytics": "api_analytics",
    "cuda-python": "cuda",
    "open_clip_torch": "open_clip",
    "python-multipart": "multipart",
}
logger = logging.getLogger(__name__)


class NoModuleSpecFound(Exception):
    "Exception raised when no module spec is found"


def install_requirements(path_to_requirements: str = "requirements.txt"):
    "Check if requirements are installed, if not, install them"

    with open(path_to_requirements, encoding="utf-8", mode="r") as f:
        requirements = {}
        for line in [r.strip() for r in f.read().splitlines()]:
            split = line.split(";")
            i: str = split[0].strip()

            if len(split) > 1:
                check = split[1].strip()
            else:
                check = ""

            if check == 'platform_system == "Linux"':
                logger.debug("Install check for Linux only")
                if platform.system() != "Linux":
                    continue

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
                # Skip extra commands for pip and comments
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

        except ImportError as e:
            logger.debug(
                f"Installing requirements: {path_to_requirements}, because: {e} ({e.__class__.__name__})"
            )
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


def install_iree():
    "Install components necessary for IREE compilation and inference"

    if (
        not is_installed("iree-compiler")
        or not is_installed("iree-runtime")
        or not is_installed("torch-mlir")
    ):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "iree-compiler",
                "iree-runtime",
                "git+https://github.com/llvm/torch-mlir.git",
                "-f",
                "https://nod-ai.github.io/SHARK-Runtime/pip-release-links.html",
            ]
        )


@dataclass
class PytorchDistribution:
    "Dataclass that holds information about a pytorch distribution"
    windows_supported: bool
    name: str
    check_command: Union[str, List[str]]
    success_message: str
    install_command: Union[str, List[str], List[List[str]]]


# Order MATTERS!
_pytorch_distributions = [
    PytorchDistribution(
        windows_supported=False,
        name="rocm",
        check_command="rocminfo",
        success_message="ROCmInfo success, assuming user has AMD GPU",
        install_command=[
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.0.0",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/rocm5.4.2",
        ],
    ),
    PytorchDistribution(
        windows_supported=True,
        name="cuda",
        check_command="nvidia-smi",
        success_message="NVidia-SMI success, assuming user has an NVIDIA GPU",
        install_command=[
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.0.0",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ],
    ),
    PytorchDistribution(
        windows_supported=False,
        name="intel",
        check_command=["test", "-f", '"/etc/OpenCL/vendors/intel.icd"'],
        success_message="Intel check success, assuming user has an Intel (i)GPU",
        install_command=[
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==1.13.0a0",
            "torchvision==0.14.1a0",
            "intel_extension_for_pytorch==1.13.120+xpu",
            "-f",
            "https://developer.intel.com/ipex-whl-stable-xpu",
        ],
    ),
    PytorchDistribution(
        windows_supported=False,
        name="cpu",
        check_command="echo a",
        success_message="No GPU detected, assuming user doesn't have one",
        install_command=[
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "intel_extension_for_pytorch",
        ],
    ),
    PytorchDistribution(
        windows_supported=True,
        name="directml",
        check_command="echo a",
        success_message="No GPU detected, assuming user needs DirectML",
        install_command=[
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torch-directml",
        ],
    ),
    PytorchDistribution(
        windows_supported=True,
        name="vulkan",
        check_command="vulkaninfo",
        success_message="Vulkan check success, assuming user has a Vulkan capable GPU",
        install_command=[
            ["git", "clone", "https://github.com/pytorch/pytorch.git"],
            [
                "USE_VULKAN=1",
                "USE_VULKAN_SHADERC_RUNTIME=1",
                "USE_VULKAN_WRAPPER=0",
                "USE_CUDA=0",
                sys.executable,
                "pytorch/setup.py",
                "install",
            ],
            [sys.executable, "-m", "pip", "install", "torchvision"],
        ],
    ),
]


def install_pytorch(force_distribution: int = -1):
    "Install necessary requirements for inference"

    # Install pytorch
    if not is_installed("torch") or not is_installed("torchvision"):
        if isinstance(force_distribution, int):
            forced_distribution = (
                _pytorch_distributions[force_distribution]
                if -1 < force_distribution < len(_pytorch_distributions)
                else None
            )
        else:
            forced_distribution = [
                x
                for x in _pytorch_distributions
                if x.name == force_distribution.lower()
            ][0]
        logger.info("Installing PyTorch")
        if platform.system() == "Darwin":
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchvision"]
            )
        else:
            for c in _pytorch_distributions:
                if (
                    (c.windows_supported if platform.system() == "Windows" else True)
                    and (
                        (
                            subprocess.run(  # pylint: disable=subprocess-run-check
                                c.check_command,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                shell=True,
                            ).returncode
                            == 0
                        )
                    )
                ) or c == forced_distribution:
                    logger.info(c.success_message)
                    if isinstance(c.install_command[0], list):
                        for cmd in c.install_command:
                            subprocess.check_call(cmd)
                    else:
                        subprocess.check_call(c.install_command)  # type: ignore
                    break

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
            raise NoModuleSpecFound

        if version is not None:
            try:
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
            except PackageNotFoundError:
                logger.debug(
                    f"Version metadata not found for {package}, skipping version check"
                )
        else:
            logger.debug(f"Package {package} - ok")

    except AssertionError:
        logger.debug(
            f"Package {package} - found - incorrect version - {version} - {importlib.metadata.version(package)}"
        )
        return False

    except NoModuleSpecFound:
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
