from argparse import ArgumentParser

from safetensors import safe_open
from safetensors.torch import save_file

parser = ArgumentParser()
parser.add_argument("input", type=str, help="Input file")
parser.add_argument("output", type=str, help="Output file")
args = parser.parse_args()

# Load the model
model = safe_open(args.input, framework="pt")
tensors = {}
for key in model.keys():
    tensors[key] = model.get_tensor(key)

# Remove broken tensors
del tensors["cond_stage_model.logit_scale"]
del tensors["cond_stage_model.text_projection"]

# Save the fixed model
save_file(tensors, args.output)
