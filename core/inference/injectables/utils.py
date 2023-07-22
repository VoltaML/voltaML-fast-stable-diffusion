from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, List, Tuple, Any, Type

import torch


class HookObject(ABC):
    "Module containing information on this subset of injectables."

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        name: str,
        prompt_key: str,
        module_class: Type,
        targets: List[str],
        default_attributes: List[Tuple[str, Any]] = [],
    ) -> None:
        self.name = name
        self.prompt_key = prompt_key
        self.module_type = module_class
        self.targets = targets
        self.default_attributes = default_attributes

        self.containers: Dict[str, module_class] = {}

    def unload(self, file: Union[Path, str]):
        "Unload a modifier file."
        if not isinstance(file, Path):
            file = Path(file)
        del self.containers[file.name]

    @abstractmethod
    def load(
        self,
        name: str,
        state_dict: Dict[str, torch.nn.Module],
        modules: Dict[str, torch.nn.Module],
    ) -> Any:
        "Load a modifier file."

    @abstractmethod
    def apply_hooks(self, p: torch.nn.Module) -> None:
        "Apply weights of this modifier to the module given in argument `p`."
