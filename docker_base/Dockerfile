FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential cmake \
         libopenblas-dev liblapack-dev \
         python3 python3-dev python3-pip \
         git \
         ca-certificates && \
     apt-get clean && \
     rm -rf /var/lib/apt/lists/*


RUN pip3 install -U pip
RUN pip3 install numpy setuptools


# install pytorch from github
RUN mkdir -p /usr/src/libs/pytorch && \
    cd /usr/src/libs/pytorch && \
    git clone --recursive https://github.com/pytorch/pytorch.git . && \
    git checkout dd5c195646b941d3e20a72847ac48c41e272b8b2 && \
    pip3 install -r requirements.txt && \
    python3 setup.py install && \
    rm -rf /usr/src/libs/pytorch
#RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl

RUN mkdir -p /usr/src/app
ADD . /usr/src/app
WORKDIR /usr/src/app

ENV PYTHONPATH /usr/src/app
CMD ["python3", "main.py"]
