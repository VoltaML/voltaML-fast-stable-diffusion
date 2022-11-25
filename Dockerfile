FROM nvcr.io/nvidia/tensorrt:22.11-py3


RUN python3 -m pip --no-cache-dir install --upgrade pip

WORKDIR /code

RUN pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

COPY . /code/

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


EXPOSE 5000

CMD [“/bin/bash”] 


