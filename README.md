# First-Order-Motion Realtime

An app to make realtime access to first-order-motion approachable.

![Demo with Toast Man](./git-assets/toastman.gif)
![Demo with Frodo Baggins](./git-assets/frodo.gif)

based on https://github.com/AliaksandrSiarohin/first-order-model and https://github.com/k0staa/real-time-example-of-first-order-motion-model-for-image-animation

## Requirements

- python 3.9.x
- [pipenv](https://pipenv.pypa.io/en/latest/install/)
- A [CUDA-Capable](https://developer.nvidia.com/cuda-gpus#compute) NVIDIA Graphics card + [drivers](https://developer.nvidia.com/cuda-downloads)

## Setup and running from source

```sh
pipenv install
pipenv run python ./app.py
```

## Building a Distributable

```sh
# get dependencies (including pyinstaller)
pipenv install

# Before proceeding, update the absolute paths in app.spec so they
# are accurate on your machine.

# package the app
pipenv run pyinstaller --noconfirm app.spec
# check it runs
./dist/first-order-motion-tk/app
# build a distributable zip
zip -r ./dist/first-order-motion-tk-linux.zip ./dist/first-order-motion-tk
```