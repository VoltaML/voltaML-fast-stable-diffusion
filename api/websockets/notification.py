from typing import Literal

from .data import Data


class Notification(Data):
    "Notification to be sent to the client"

    def __init__(
        self,
        severity: Literal["info", "warning", "error", "success"] = "info",
        title: str = "",
        message: str = "",
    ) -> None:
        self.severity = severity
        self.title = title
        self.message = message
        super().__init__(
            data={
                "severity": self.severity,
                "title": self.title,
                "message": self.message,
            },
            data_type="notification",
        )
