import tensorrt as trt
import os, sys, argparse 
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # without this, "LogicError: explicit_context_dependent failed: invalid device context - no currently active context?"
from time import time
import onnx
import torch
from diffusers.models import unet_2d_condition_onnx
import gc
import subprocess
import onnxsim
import onnxconverter_common


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="runwayml/stable-diffusion-v1-5", type=str, help="Diffusion model path.")
    parser.add_argument("--save_path", default='unet.engine', type=str, help="TensorRT saved path")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--img_size", default=(512,512),help="Unet input image size (h,w)")
    parser.add_argument("--max_seq_length", default=64,help="Maximum sequence length of input text")
    return parser.parse_args()

def convert_to_onnx(args):

    unet = unet_2d_condition_onnx.UNet2DConditionModel.from_pretrained(args.model_path,
                                            subfolder="unet",
                                            use_auth_token=True)

    if not os.path.exists('unet'):
        os.makedirs('unet')
    
    height=args.img_size[0]
    width=args.img_size[1]
    h, w = height // 8, width // 8
    check_inputs = [(torch.rand(2, 4, h, w), torch.tensor([980], dtype=torch.long), torch.rand(2, 77, 768)), 
                    (torch.rand(2, 4, h, w), torch.tensor([910], dtype=torch.long), torch.rand(2, 12, 768)), # batch change, text embed with no trunc
                    ]
    traced_model = torch.jit.trace(unet, check_inputs[0], check_inputs=[check_inputs[1]], strict=True)

    # Export the model
    torch.onnx.export(traced_model,               # model being run
                    check_inputs[0],                         # model input (or a tuple for multiple inputs)
                    "unet/model.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names =["latent_model_input", "t", "encoder_hidden_states"],
                    output_names = ['out_sample'],
                    dynamic_axes={ "latent_model_input": [0], "t": [0], "encoder_hidden_states": [0, 1]})



    new_onnx_model = onnxconverter_common.convert_float_to_float16_model_path("unet/model.onnx",keep_io_types=True)
    onnx.save(new_onnx_model, 'unet/model_fp16.onnx')
    del unet, new_onnx_model
    torch.cuda.empty_cache()
    gc.collect()

def convert_to_trt(args):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    TRT_BUILDER = trt.Builder(TRT_LOGGER)
    network = TRT_BUILDER.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)
    parse_success = onnx_parser.parse_from_file('./unet/model.onnx')
    for idx in range(onnx_parser.num_errors):
        print(onnx_parser.get_error(idx))
    if not parse_success:
        sys.exit('ONNX model parsing failed')
    config = TRT_BUILDER.create_builder_config()
    profile = TRT_BUILDER.create_optimization_profile() 

    latents_shape = (args.batch_size*2, 4, args.img_size[0] // 8, args.img_size[1] // 8)
    embed_shape = (args.batch_size*2, args.max_seq_length, 768)
    timestep_shape = (args.batch_size,)

    
    profile.set_shape("latent_model_input", latents_shape, latents_shape, latents_shape) 
    profile.set_shape("encoder_hidden_states", embed_shape, embed_shape, embed_shape) 
    profile.set_shape("t", timestep_shape, timestep_shape, timestep_shape) 
    config.add_optimization_profile(profile)

    #config.max_workspace_size = 4096 * (1 << 20)
    config.set_flag(trt.BuilderFlag.FP16)
    serialized_engine = TRT_BUILDER.build_serialized_network(network, config)
            
    ## save TRT engine
    with open(args.save_path, 'wb') as f:
        f.write(serialized_engine)
    print(f'Engine is saved to {args.save_path}')
    

if __name__ == "__main__":
    args = get_args()
    convert_to_onnx(args)
    sys.stdout.flush()
    ## Command to convert onnx to tensorrt
    convert_to_trt(args)
