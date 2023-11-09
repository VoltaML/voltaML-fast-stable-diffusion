from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class FrontendConfig:
    "Configuration for the frontend"

    theme: Union[Literal["dark", "light"], str] = "dark"
    enable_theme_editor: bool = False
    image_browser_columns: int = 5
    on_change_timer: int = 0
