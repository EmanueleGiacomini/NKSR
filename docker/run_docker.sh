#!/bin/bash

xhost local:root && docker run -it --rm -e SDL_VIDEODRIVER=x11 -e DISPLAY=$DISPLAY --env='DISPLAY' \
  --ipc host --privileged --network host -p 8080:8081 --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/rvp-00/source/eg/NKSR/:/workspace \
  -v /media/rvp-00/DATA1/datasets/:/storage \
  nksr:localbuild xfce4-terminal --title=NKSR
