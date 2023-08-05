from dataclasses import dataclass
from typing import Literal


@dataclass
class FrontendConfig:
    "Configuration for the frontend"

    theme: Literal["dark", "light"] = "dark"
    enable_theme_editor: bool = False
    image_browser_columns: int = 5
    on_change_timer: int = 0
    nsfw_ok_threshold: int = 0