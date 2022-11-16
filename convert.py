import tensorrt as trt
import os, sys, argparse 
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # without this, "LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
from time import time
import onnx
import torch
from diffusers import UNet2DConditionModel
import gc
import subprocess
import onnxsim

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="runwayml/stable-diffusion-v1-5", type=str, help="Diffusion model path.")
    parser.add_argument("--save_path", default='unet_3.engine', type=str, help="TensorRT saved path")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--img_size", default=(512,512),help="Unet input image size (h,w)")
    parser.add_argument("--max_seq_length", default=64,help="Maximum sequence length of input text")
    return parser.parse_args()

def convert_to_onnx(model="runwayml/stable-diffusion-v1-5"):

    unet = UNet2DConditionModel.from_pretrained(model,
                                            torch_dtype=torch.float16,
                                            subfolder="unet",
                                            use_auth_token=True)

    unet.cuda()

    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = torch.randn(2,4,64,64, dtype=torch.half, device='cuda'), torch.randn(1, dtype=torch.half, device='cuda'), torch.randn(2, 77, 768, dtype=torch.half, device='cuda')

        # Export the model
        torch.onnx.export(unet,               # model being run
                        inputs,                         # model input (or a tuple for multiple inputs)
                        "tmp.onnx",   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=16,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names =["latent_model_input", "t", "encoder_hidden_states"],
                        output_names = ['out_sample'],
                        dynamic_axes={ "latent_model_input": [0], "t": [0], "encoder_hidden_states": [0, 1]})
    
    del unet

    f = 'tmp.onnx'
    print('-------- Simplify ONNX ---------------')
    # Checks
    # model_onnx = onnx.load(f)  # load onnx model
    # onnx.checker.check_model(model_onnx)  # check onnx model
    model_onnx, check = onnxsim.simplify(f)
    onnx.save(model_onnx, 'tmp_sim.onnx')
    os.remove('tmp.onnx')
    del model_onnx
    model_onnx = None
    torch.cuda.empty_cache()
    gc.collect()

    

if __name__ == "__main__":
    args = get_args()
    convert_to_onnx(model="runwayml/stable-diffusion-v1-5")

    sys.stdout.flush()
    #os.execl(sys.executable, 'python', __file__, *sys.argv[1:])

    ## Command to convert onnx to tensorrt
    command = "trtexec --onnx=./tmp_sim.onnx --saveEngine=unet.engine --fp16 --optShapes=latent_model_input:2x4x64x64,t:1,encoder_hidden_states:2x64x768 --workspace=11245MiB"
    subprocess.run(command, shell=True, check=True)
    os.remove('tmp_sim.onnx')
