from typing import Literal

from .data import Data


class Notification(Data):
    "Notification to be sent to the client"

    def __init__(
        self,
        severity: Literal["info", "warning", "error", "success"] = "info",
        title: str = "",
        message: str = "",
        timeout: int = 10_000,
    ) -> None:
        self.severity = severity
        self.title = title
        self.message = message
        self.timeout = timeout
        super().__init__(
            data={
                "severity": self.severity,
                "title": self.title,
                "message": self.message,
                "timeout": self.timeout,
            },
            data_type="notification",
        )
