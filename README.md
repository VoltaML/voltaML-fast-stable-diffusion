

<p align="center">
  <img width="1000" height="500" src="https://user-images.githubusercontent.com/107309002/207094372-9aacc79e-7731-41ea-8d77-29d5ce75167f.png">
</p>


<h1 align="center">🔥 🔥 voltaML-fast-stable-diffusion webUI 🔥 🔥 </h1>

<p align="center">
  <b> Accelerate your machine learning and deep learning models by upto 10X </b> 
</p>

###                                                   

<div align="center">
<a href="https://discord.gg/pY5SVyHmWm"> <img src="https://dcbadge.vercel.app/api/server/pY5SVyHmWm" /> </a> 
</div>


Lightweight library to accelerate Stable-Diffusion, Dreambooth into fastest inference models with **WebUI single click or single line of code**.

<h1 align="center"> Setup webUI </h3>

![Screenshot from 2022-12-12 11-19-09](https://user-images.githubusercontent.com/107309002/206970939-f827f7b9-6966-41c1-a2aa-3ed18e73d399.png)

![Screenshot from 2022-12-12 11-36-37](https://user-images.githubusercontent.com/107309002/206972269-1223c567-3df8-41c5-a7b3-f31e544b98aa.png)


#### Docker setup (if required)
Setup docker on Ubuntu using [these intructions](https://docs.docker.com/engine/install/ubuntu/).

Setup docker on Windows using [these intructions](https://docs.docker.com/desktop/install/windows-install/)


### Launch voltaML container
```
sudo docker run --gpus=all -v $pwd/engine:/workspace/volta_stable_diffusion/engine -it -p "8800:8800" voltaml/volta_stable_diffusion:v0.2
```
⚠️ You need to mount a local volume to save your work onto your system. Or else the work will be deleted once you exit the container </br>
⚠️ To save your work in the container itself, you have to commit the container and then exit the container.

### How to use webUI 
 1. Once you launch the container, a flask app will run and copy/paste the url to run the webUI on your local host.
 ![Screenshot from 2022-12-12 12-36-01](https://user-images.githubusercontent.com/107309002/206982082-ee498781-9e6d-4b80-a652-2e4e29a2835e.png)

 2. There are two backends to run the SD on, PyTorch and TensorRT (fastest version)
 3. To run on PyTorch inference, you have to select the model, the model will be downloaded (which will take a few mins) into the container and the inference will be displayed. Downloaded models will be shown as below
![download_sd](https://user-images.githubusercontent.com/107309002/206983689-5f40f446-426b-45b7-88fa-db224099dd8e.png)
 4. To run TensoRT inference, go to the Accelerate tab, pick a model from our model hub and click on the accelerate button. <br/>
![Screenshot from 2022-12-12 13-17-23](https://user-images.githubusercontent.com/107309002/206989892-6f04dbdf-312b-41b3-bb69-684610659fae.png)
 5. Once acceleration is done, the model will show up in your TensorRT drop down menu.
 6. Switch your backend to TensorRT, select the model and enjoy the fastest outputs 🚀🚀 

## Benchmark
```
python3 volta_infer.py --backend='TRT' --benchmark
```
The below benchmarks have been done for generating a 512x512 image, batch size 1 for 50 iterations.

| Model          | T4 (it/s)      | A10 (it/s)      | A100 (it/s)       |
|----------------|--------------|----------------|----------------|
| PyTorch        |     4.3      | 8.8            | 15.1           |
| Flash attention xformers| 5.5 | 15.6            |27.5            |
| AITemplate     | Not supported | 26.7               | 55|
| VoltaML(TRT-Flash)   |     11.4      | 29.2            | 62.8           |

 
### ⚠️ ‼️ Warnings/Caveats

**This is v0.1 of the product. Things might break. A lot of improvements are on the way, so please bear with us.**

1. This will only work for NVIDIA GPUs with compute capability > 7.5
2. Cards with less than 12GB VRAM will have issues with acceleration, due to high memory required for the conversions. We're working on resolving these in our next release.  
3. While the model is accelerating, **no other functionality will work since the GPU will be fully occupied**
