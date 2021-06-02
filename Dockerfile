FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip python3.7 ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

COPY . /app/
WORKDIR /app

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip --version

RUN python3.7 -m pip install \
  git+https://github.com/1adrianb/face-alignment \
  -r requirements.txt

RUN python3.7 -m pip install notebook ffmpeg-python gdown

EXPOSE 8888

CMD ["jupyter notebook --no-browser --port 8888 --ip=* --allow-root"]
