from typing import Any, Dict


class Data:
    def __init__(self, data: dict, type: str):
        self.data = data
        self.type = type

    def to_json(self) -> Dict[str, Any]:
        return {"type": self.type, "data": self.data}
