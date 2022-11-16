import torch
from tqdm import tqdm
from PIL import Image
from trt_model import TRTModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from diffusers import PNDMScheduler
from torch import autocast
import argparse
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="mdjrny-v4 style e cyborg woman| with a visible detailed brain| muscles cable wires| detailed cyberpunk background with neon lights| biopunk| cybernetic| unreal engine| CGI | ultra detailed| 4k", help="input prompt")
    parser.add_argument("--trt_unet_save_path", default="./unet.engine", type=str, help="TensorRT unet saved path")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--img_size", default=(512,512),help="Unet input image size (h,w)")
    parser.add_argument("--max_seq_length", default=64,help="Maximum sequence length of input text")
    parser.add_argument("--benchmark", action="store_true",help="Running benchmark by average num iteration")
    parser.add_argument("--n_iters", default=50, help="Running benchmark by average num iteration")

    return parser.parse_args()

class TrtDiffusionModel():
    def __init__(self, args):
        self.device = torch.device("cuda")
        self.unet = TRTModel(args.trt_unet_save_path)
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            use_auth_token=True).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="tokenizer",
            use_auth_token=True)
        self.text_encoder = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="text_encoder",
            use_auth_token=True).to(self.device)
        self.scheduler = PNDMScheduler.from_config("scheduler")

    def predict(
        self, 
        prompts,
        # seed = 1948952866,
        num_inference_steps = 50,
        height = 512,
        width = 512,
        max_seq_length = 64
    ):
        guidance_scale = 15
        batch_size = 1
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_seq_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8)).to(self.device)
        self.scheduler.set_timesteps(num_inference_steps)

        torch.backends.cudnn.benchmark = True

        # latents = latents * self.scheduler.sigmas[0]
        with torch.inference_mode(), autocast("cuda"):
            for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                latent_model_input = torch.cat([latents] * 2)
                # sigma = self.scheduler.sigmas[i]
                # latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                # predict the noise residual
                inputs = [latent_model_input, torch.tensor([t]).to(self.device),text_embeddings]
                # import ipdb; ipdb.set_trace()
                noise_pred, duration = self.unet(inputs, timing=True)
                noise_pred  = torch.reshape(noise_pred[0],(batch_size*2,4,64,64))

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.cuda(), t, latents)["prev_sample"]

            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
        return image

if __name__ == "__main__":
    args = get_args()
    model = TrtDiffusionModel(args)
    if args.benchmark:
        n_iters = args.n_iters
        #warm up
        for i in range(5):
            image = model.predict(
                prompts = args.prompt,
                num_inference_steps = 50,
                height = args.img_size[0],
                width = args.img_size[1],
                max_seq_length = args.max_seq_length
            )
    else:
        n_iters = 1

    start = time.time()
    for i in tqdm(range(n_iters)):
        image = model.predict(
            prompts = args.prompt,
            num_inference_steps = 50,
            height = args.img_size[0],
            width = args.img_size[1],
            max_seq_length = args.max_seq_length
        )
    end = time.time()

    #total_time = end - start

    #print(total_time)

    if args.benchmark:
        print("Average inference time is: ",(end-start)/n_iters)
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save('image_generated.png')