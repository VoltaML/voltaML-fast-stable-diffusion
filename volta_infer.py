import argparse
from pytorch_model import load_model, inference
from pathlib import Path
import uuid
import torch
from tqdm import tqdm
from PIL import Image
from trt_model import TRTModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from diffusers import PNDMScheduler
from torch import autocast
import time




def get_args():
    """Configure argparser

    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Super Mario learning to fly in an airport, Painting by Leonardo Da Vinci", help="input prompt")
    parser.add_argument("--img_height", type=int, default=512, help="The height in pixels of the generated image.")
    parser.add_argument("--img_width", type=int, default=512, help="The width in pixels of the generated image.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="The number of images to generate per prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Seed to make generation deterministic")
    parser.add_argument("--saving_path", type=str, default="generated_images", help="Directory where the generated images will be saved")
    parser.add_argument("--backend", type=str, default="PT", help="PT , TRT")
    parser.add_argument("--trt_unet_save_path", default="./unet.engine", type=str, help="TensorRT unet saved path")
    parser.add_argument("--benchmark", action="store_true",help="Running benchmark by average num iteration")
    parser.add_argument("--max_seq_length", default=64,help="Maximum sequence length of input text")
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
        
        latents_shape = (1, 4, height // 8, width // 8)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
                latents_shape, generator=generator, device=self.device
            )
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

    # Create directory to save images if it does not exist
    saving_path = Path(args.saving_path)
    if not saving_path.exists():
        saving_path.mkdir(exist_ok=True, parents=True)
        
    if args.backend == 'PT':
    
        print("[+] Loading the model")
        model = load_model()
        print("[+] Model loaded")

        print("[+] Generating images...")
        # PIL images
        images, time = inference(
            model=model,
            prompt=args.prompt,
            img_height=args.img_height,
            img_width=args.img_width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
            seed=args.seed,
            return_time=True
        )

        print("[+] Time needed to generate the images: {} seconds".format(time))

        # Save PIL images with a random name
        for img in images:
            img.save('{}/{}.png'.format(
                saving_path.as_posix(),
                uuid.uuid4()
            ))

        print("[+] Images saved in the following path: {}".format(saving_path.as_posix()))
    
    elif args.backend == 'TRT':
        
        model = TrtDiffusionModel(args)
        if args.benchmark:
            n_iters = args.num_inference_steps
            #warm up
            for i in range(5):
                image = model.predict(
                    prompts = args.prompt,
                    num_inference_steps = 50,
                    height = args.img_height,
                    width = args.img_width,
                    max_seq_length = args.max_seq_length
                )
        else:
            n_iters = 1

        start = time.time()
        for i in tqdm(range(n_iters)):
            image = model.predict(
                prompts = args.prompt,
                num_inference_steps = 50,
                height = args.img_height,
                width = args.img_width,
                max_seq_length = args.max_seq_length
            )
        end = time.time()

        # total_time = end - start
        # print("[+] Time needed to generate the images: {} seconds".format(total_time))

        if args.benchmark:
            print("Average inference time is: ",(end-start)/n_iters)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        pil_images[0].save('image_generated.png')
        
    else:
        print('----- Please chose appropriate backend. PT for PyTorch / TRT for TensorRT ------')
