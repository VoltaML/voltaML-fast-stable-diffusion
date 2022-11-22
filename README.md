![Screenshot from 2022-11-22 15-29-39](https://user-images.githubusercontent.com/107309002/203284627-fa180962-75b1-41dd-83a7-124b74a1fcdf.png)


# ⚡voltaML-fast-stable-diffusion 🔥 🔥 
Lightweight library to accelerate Stable-Diffusion, Dreambooth into fastest inference models with one single line of code.

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

Requirements:
* transformers <br/>
* diffusers <br/>
* torch==1.12.1+cu116 <br/>
* scipy <br/>
* uvicorn <br/>
* pydantic <br/>
* fastapi <br/>
* pycuda <br/>
* huggingface_hub <br/>
* onnxsim <br/>
* onnxruntime <br/>
* nvidia-tensorrt==8.4.2.4 <br/>
* onnxconverter_common <br/>
* ftfy <br/>
* spacy <br/>
* accelerate <br/>


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

| Model          | T4 (ms)      | A100 (ms)      | A10 (ms)       |
|----------------|--------------|----------------|----------------|
| PyTorch        |     14.1      | 4.4            | 6.6           |
| VoltaML(TRT)   |     8.9      | 1.8            | 4.2           |


![Screenshot from 2022-11-22 09-36-04](https://user-images.githubusercontent.com/107309002/203323895-07f2cec6-d745-4955-9605-e8a4f6b3f613.png)
![Screenshot from 2022-11-22 09-36-31](https://user-images.githubusercontent.com/107309002/203323901-b12dd1ba-044d-4b2a-89aa-f04e418d949a.png)
![Screenshot from 2022-11-22 09-37-25](https://user-images.githubusercontent.com/107309002/203323904-9bfe698b-0469-4da5-bac0-7d437c805607.png)
![Screenshot from 2022-11-22 09-37-45](https://user-images.githubusercontent.com/107309002/203323906-11262ba3-d5f8-47f3-80e8-e970c3af93a1.png)
