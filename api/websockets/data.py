from typing import Any


class Data:
    def __init__(self, data: dict, type: str):
        self.data = data
        self.type = type

    def to_json(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data}
