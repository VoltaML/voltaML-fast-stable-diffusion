![Screenshot from 2022-11-22 15-29-39](https://user-images.githubusercontent.com/107309002/203284627-fa180962-75b1-41dd-83a7-124b74a1fcdf.png)

## ⚡voltaML-fast-stable-diffusion 🔥 🔥 


Lightweight library to accelerate Stable-Diffusion, Dreambooth into fastest inference models with **one single line of code**.


<div align="center">
<a href="https://discord.gg/pY5SVyHmWm"> <img src="https://dcbadge.vercel.app/api/server/pY5SVyHmWm" /> </a> 
</div>

## Installation

### voltaML Docker Container 🐳
````
docker pull voltaml/volta_diffusion:v0.1
docker run -it --gpus=all -p "8888:8888" voltaml/volta_diffusion:v0.1 \ 
        jupyter lab --port=8888 --no-browser --ip 0.0.0.0 --allow-root
        
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion.git
cd voltaML-fast-stable-diffusion
pip3 install -r requirements.txt
````

### Own setup:

Requirements: Please refer to the requirements.txt file to set it up on your own environment.

It is recommended to use our voltaml/volta_diffusion container or NVIDIA TensorRT 22.08-py3 container

## Usage

### Accelerate
```
python3 volta_accelerate.py --model='runwayml/stable-diffusion-v1-5' # your model path/ hugging face name
```

### Inference

**For TensorRT**
```
python3 volta_infer.py --backend='TRT' --prompt='a gigantic robotic bipedal dinosaur, highly detailed, photorealistic, digital painting, artstation, concept art, sharp focus, illustration, art by greg rutkowski and alphonse mucha'
```
**For PyTorch**
```
python3 volta_infer.py --backend='PT' --prompt='a gigantic robotic bipedal dinosaur, highly detailed, photorealistic, digital painting, artstation, concept art, sharp focus, illustration, art by greg rutkowski and alphonse mucha'
```
## Benchmark
```
python3 volta_infer.py --backend='TRT' --benchmark
```
The below benchmarks have been done for generating a 512x512 image, batch size 1 for 50 iterations.

| Model          | T4 (ms)      | A100 (ms)      | A10 (ms)       |
|----------------|--------------|----------------|----------------|
| PyTorch        |     14.1      | 4.4            | 6.6           |
| VoltaML(TRT)   |     8.9      | 1.8            | 4.2           |


![Screenshot from 2022-11-22 09-36-04](https://user-images.githubusercontent.com/107309002/203323895-07f2cec6-d745-4955-9605-e8a4f6b3f613.png)
![Screenshot from 2022-11-22 09-36-31](https://user-images.githubusercontent.com/107309002/203323901-b12dd1ba-044d-4b2a-89aa-f04e418d949a.png)
![Screenshot from 2022-11-22 09-37-25](https://user-images.githubusercontent.com/107309002/203323904-9bfe698b-0469-4da5-bac0-7d437c805607.png)
![Screenshot from 2022-11-22 09-37-45](https://user-images.githubusercontent.com/107309002/203323906-11262ba3-d5f8-47f3-80e8-e970c3af93a1.png)

## To-Do:
* Integrate Flash-attention
* Integrate AITemplate
* Try Flash-attention with TensorRT

## Contribution:
We invite the open source community to contribute and help us better voltaML. Please check out our [contribution guide](https://github.com/VoltaML/voltaML-fast-stable-diffusion/blob/main/CONTRIBUTION.md)

## References
* https://www.photoroom.com/tech/stable-diffusion-25-percent-faster-and-save-seconds/ </br>
* https://github.com/kamalkraj/stable-diffusion-tritonserver </br>
* https://github.com/luohao123/gaintmodels </br>
* https://github.com/stochasticai/x-stable-diffusion
