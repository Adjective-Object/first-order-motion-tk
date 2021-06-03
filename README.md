# First-Order-Motion Realtime

An app to make realtime access to first-order-motion approachable.

![Demo with Toast Man](./git-assets/toastman.gif)
![Demo with Frodo Baggins](./git-assets/frodo.gif)

based on https://github.com/AliaksandrSiarohin/first-order-model and https://github.com/k0staa/real-time-example-of-first-order-motion-model-for-image-animation

## Requirements

- python 3.9.x
- [pipenv](https://pipenv.pypa.io/en/latest/install/)
- A [CUDA-Capable](https://developer.nvidia.com/cuda-gpus#compute) NVIDIA Graphics card + [drivers](https://developer.nvidia.com/cuda-downloads)

## Controls

- `relative_movement`: When checked, this toggle poses target face based on the changes in the pose of your own face makes. When not checked, the target face will be mapped directly onto the position / shape of your face.
- `relative_jacobian`: When checked, if relative_movement is active, applies distortions to the model's impression of your face relative to the starting position to the base face. When `relative_movement` is turned on but `relative_jacobian` is turned off, the pose will be reconstructed relatively, but the model will try to reconstruct the face based on the shape of your face.
- `adapt_movement_scale`: When checked, this will re-scale your movements based on the size of the target face. For example, if the target face is larger than yours and you rotate your head, the target face will be rotated approximately around its own neck, instead of around where _your_ neck would be.
- `camera_zoom`: controls how tightly the input image is cropped about the center
- `reset_initial_frame`: resets the initial frame used in relative_movement and relative_jacobian to the current frame of the camera.

## Setup and running from source

```sh
pipenv install --dev
pipenv run python ./app.py
```

## Building a Distributable

```sh
# get dependencies (including pyinstaller)
pipenv install --dev

# Before proceeding, update the absolute paths in app.spec so they
# are accurate on your machine.

# package the app
pipenv run pyinstaller --noconfirm app.spec
# check it runs
./dist/first-order-motion-tk/app
# build a distributable zip
zip -r ./dist/first-order-motion-tk-linux.zip ./dist/first-order-motion-tk
# or, build a distributable tar.xz if the zip is too big
tar -cf - ./dist/first-order-motion-tk | xz -4e > ./dist/first-order-motion-tk.tar.xz
```

## Profiling

```sh
pipenv run py-spy record -o profile.svg --subprocesses -- python app.py
```