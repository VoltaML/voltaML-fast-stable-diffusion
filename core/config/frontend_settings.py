from dataclasses import dataclass


@dataclass
class FrontendConfig:
    "Configuration for the frontend"

    theme: str = "dark"
    background_image_override: str = ""
    enable_theme_editor: bool = False
    image_browser_columns: int = 5
    on_change_timer: int = 0
    nsfw_ok_threshold: int = 0
    disable_analytics: bool = False
