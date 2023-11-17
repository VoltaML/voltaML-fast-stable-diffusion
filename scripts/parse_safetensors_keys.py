import json
import struct
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

parser = ArgumentParser()
parser.add_argument("file", type=Path, help="File to parse")
parser.add_argument(
    "-o", "--output", type=Path, help="Output to file instead of stdout"
)
parser.add_argument("--keys", action="store_true", help="Output only keys")
args = parser.parse_args()


def parse(file: Path) -> Union[dict, list]:
    if file.suffix == ".safetensors":
        with open(file, "rb") as f:
            ff = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(ff))
    elif file.suffix == ".ckpt":
        # We will need to load the whole model into memory
        import torch

        state_dict = torch.load(file)["state_dict"]

        all_keys = []
        for item in state_dict.items():
            all_keys.append(item[0])

        return all_keys

    elif file.is_dir():
        if not (file / "model_index.json").exists():
            raise ValueError("Unparseable folder, missing model_index.json")
        with open(file / "model_index.json", "r") as f:
            return json.loads(f.read())
    else:
        raise ValueError("Unparseable file")


parsed = parse(args.file)
output = (
    list(parsed.keys()) if args.keys and isinstance(parsed, dict) else parse(args.file)
)
if args.output:
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
else:
    print(output)
