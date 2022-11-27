![Screenshot from 2022-11-22 15-29-39](https://user-images.githubusercontent.com/107309002/203284627-fa180962-75b1-41dd-83a7-124b74a1fcdf.png)

## ⚡voltaML-fast-stable-diffusion 🔥 🔥 


Lightweight library to accelerate Stable-Diffusion, Dreambooth into fastest inference models with **one single line of code**.

<div align="center">
<a href="https://discord.gg/pY5SVyHmWm"> <img src="https://dcbadge.vercel.app/api/server/pY5SVyHmWm" /> </a> 
</div>

### **🔥[Accelerate Computer vision, NLP models etc.](https://github.com/VoltaML/voltaML) with voltaML. Upto 10X speed up in inference🔥**

## Installation

### voltaML Docker Container 🐳
````        
git clone https://github.com/VoltaML/voltaML-fast-stable-diffusion.git
cd voltaML-fast-stable-diffusion

docker build -t voltaml/volta_diffusion:v0.1 .

docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/code --rm voltaml/volta_diffusion:v0.1
````

### Own setup:

Requirements: Please refer to the requirements.txt file to set it up on your own environment.

It is recommended to use our voltaml/volta_diffusion container or NVIDIA TensorRT 22.08-py3 container

## Usage

### Huggingface Login
Login into your huggingface account through the terminal
```
huggingface-cli login
Token: #enter your huggingface token
```
### Accelerate
```
bash optimize.sh --model='runwayml/stable-diffusion-v1-5' # your model path/ hugging face name
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


![diffusion posts](https://user-images.githubusercontent.com/107309002/203910224-e4e89fe5-5929-4e5e-ac8d-4f126fc5c273.jpg)
![diffusion posts 1](https://user-images.githubusercontent.com/107309002/203910230-f83eda45-eb85-48a2-b5c8-e4f3ec8c21dd.jpg)
![diffusion posts 3](https://user-images.githubusercontent.com/107309002/203910233-79991ee4-24e1-4ac0-b0b2-d41543f75cef.jpg)
![diffusion posts 4](https://user-images.githubusercontent.com/107309002/203910349-1168695b-816f-4d35-9fa7-7e0331816eeb.jpg)

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
