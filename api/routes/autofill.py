import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter

router = APIRouter(tags=["autofill"])
logger = logging.getLogger(__name__)


@router.get("/")
def get_autofill_list() -> List[str]:
    "Gathers and returns all words from the prompt autofill files"

    autofill_folder = Path("data/autofill")

    words = []

    logger.debug(f"Looking for autofill files in {autofill_folder}")
    logger.debug(f"Found {list(autofill_folder.iterdir())} files")

    for file in autofill_folder.iterdir():
        if file.is_file():
            if file.suffix == ".txt":
                logger.debug(f"Found autofill file: {file}")
                with open(file, "r", encoding="utf-8") as f:
                    words.extend(f.read().splitlines())

    return list(set(words))
