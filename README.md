![Screenshot from 2022-11-22 15-29-39](https://user-images.githubusercontent.com/107309002/206966210-7e9e57a0-0184-431b-8d28-304c46bce02b.png)


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

 3. 

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
