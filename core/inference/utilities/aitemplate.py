from typing import Any, Union
from pathlib import Path
import os


def init_ait_module(
    model_name: str,
    workdir: Union[str, Path],
) -> Any:
    "Initialize a new AITemplate Module object"
    from aitemplate.compiler import Model

    mod = Model(os.path.join(workdir, model_name, "test.so"))
    return mod
