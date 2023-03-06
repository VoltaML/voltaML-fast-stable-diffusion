from typing import Any, Dict, List, Union


class Data:
    "Data to be sent to the client"

    def __init__(self, data: Union[Dict[Any, Any], List[Any]], data_type: str):
        self.data = data
        self.type = data_type

    def to_json(self) -> Union[Dict[str, Any], List[Any]]:
        "Converts the data to a JSON object"

        return {"type": self.type, "data": self.data}
