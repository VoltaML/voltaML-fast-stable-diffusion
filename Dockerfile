FROM nvcr.io/nvidia/pytorch:22.11-py3

RUN python3 -m pip --no-cache-dir install --upgrade pip

WORKDIR /code

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt
