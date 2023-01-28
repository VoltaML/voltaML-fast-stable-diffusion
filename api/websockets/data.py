from typing import Any, Dict


class Data:
    "Data to be sent to the client"

    def __init__(self, data: dict, data_type: str):
        self.data = data
        self.type = data_type

    def to_json(self) -> Dict[str, Any]:
        "Converts the data to a JSON object"

        return {"type": self.type, "data": self.data}
