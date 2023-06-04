# Old VoltaML

This section is for the old VoltaML, which is no longer maintained. It is kept here for anyone who wants to try TensorRT.

<hr>

![Screenshot from 2022-12-12 11-19-09](https://user-images.githubusercontent.com/107309002/206970939-f827f7b9-6966-41c1-a2aa-3ed18e73d399.png)

![Screenshot from 2022-12-12 11-36-37](https://user-images.githubusercontent.com/107309002/206972269-1223c567-3df8-41c5-a7b3-f31e544b98aa.png)

### Docker setup (if required)

Setup docker on Ubuntu using [these instructions](https://docs.docker.com/engine/install/ubuntu/).

Setup docker on Windows using [these instructions](https://docs.docker.com/desktop/install/windows-install/)

### Launch voltaML container

Download the [docker-compose.yml](https://raw.githubusercontent.com/VoltaML/voltaML-fast-stable-diffusion/old/docker-compose.yml) file from this repo.

⚠️ Linux: Open it in a text editor and change the path of the output folder. It was configured for Windows only.

```yaml
output:
  driver: local
  driver_opts:
    type: none
    device: C:\voltaml\output # this line
    o: bind
```

**Then, open a terminal in that folder and run the following command**

#### Linux

```bash
sudo docker-compose up
```

#### Windows

```bash
docker-compose up
```

### How to use webUI

1.  Once you launch the container, a flask app will run and copy/paste the url to run the webUI on your local host.
    ![Screenshot from 2022-12-12 12-36-01](https://user-images.githubusercontent.com/107309002/206982082-ee498781-9e6d-4b80-a652-2e4e29a2835e.png)

2.  There are two backends to run the SD on, PyTorch and TensorRT (fastest version by NVIDIA).
3.  To run on PyTorch inference, you have to select the model, the model will be downloaded (which will take a few mins) into the container and the inference will be displayed. Downloaded models will be shown as below
    ![download_sd](https://user-images.githubusercontent.com/107309002/206983689-5f40f446-426b-45b7-88fa-db224099dd8e.png)
4.  To run TensorRT inference, go to the Accelerate tab, pick a model from our model hub and click on the accelerate button. <br/>
    ![Screenshot from 2022-12-12 13-17-23](https://user-images.githubusercontent.com/107309002/206989892-6f04dbdf-312b-41b3-bb69-684610659fae.png)
5.  Once acceleration is done, the model will show up in your TensorRT drop down menu.
6.  Switch your backend to TensorRT, select the model and enjoy the fastest outputs 🚀🚀

## Benchmark

The below benchmarks have been done for generating a 512x512 image, batch size 1 for 50 iterations.

| Model                    | T4 (it/s)     | A10 (it/s) | A100 (it/s) | 4090 (it/s) | 3090 (it/s) | 2080Ti (it/s) |
| ------------------------ | ------------- | ---------- | ----------- | ----------- | ----------- | ------------- |
| PyTorch                  | 4.3           | 8.8        | 15.1        | 19          | 11          | 8             |
| Flash attention xformers | 5.5           | 15.6       | 27.5        | 28          | 15.7        | N/A           |
| AITemplate               | Not supported | 26.7       | 55          | 60          | N/A         | Not supported |
| VoltaML(TRT-Flash)       | 11.4          | 29.2       | 62.8        | 85          | 44.7        | 26.2          |

### ⚠️ ‼️ Warnings/Caveats

**This is v0.1 of the product. Things might break. A lot of improvements are on the way, so please bear with us.**

1. This will only work for NVIDIA GPUs with compute capability > 7.5.
2. Cards with less than 12GB VRAM will have issues with acceleration, due to high memory required for the conversions. We're working on resolving these in our next release.
3. While the model is accelerating, **no other functionality will work since the GPU will be fully occupied**
