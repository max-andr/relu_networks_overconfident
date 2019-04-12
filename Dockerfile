FROM tensorflow/tensorflow:1.10.1-gpu-py3

RUN apt-get update -y
RUN apt-get install -y htop curl vim python3-tk git

RUN pip install --upgrade pip
RUN pip install torch==0.4.1 torchvision

RUN cd /
ENTRYPOINT bash