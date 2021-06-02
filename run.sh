xhost +

podman run --rm -it --device=/dev/video1:/dev/video1 \
       	--env DISPLAY=$DISPLAY \
        --env="QT_X11_NO_MITSHM=1" \
        --env="__NV_PRIME_RENDER_OFFLOAD=1" \
        --env="__GLX_VENDOR_LIBRARY_NAME=nvidia" \
        --env="__VK_LAYER_NV_optimus=NVIDIA_only" \
        -v /dev/video1:/dev/video1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro  \
        -v "${PWD}:/app" \
       	-p 8888:8888 -p 6006:6006 \
        --name first-order-model \
	    first-order-model jupyter notebook --no-browser --port 8888 --ip=* --allow-root

xhost -
