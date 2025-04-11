FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive 
ENV NVENCODE_CFLAGS="-I/usr/local/cuda/include"
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
  neovim \
  xfce4-terminal \
  wget \
  ffmpeg libsm6 libxext6 && \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
  protobuf==3.19.6 \
  mkl==2023.2.0 \
  ipython \
  tqdm \
  torch_geometric \
  rich \
  open3d \
  pandas \
  omegaconf \
  flatten-dict \
  pyntcloud \
  randomname \
  pykdtree \
  plyfile \
  typing_extensions==4.7.0


RUN python3 -m pip install \
  torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

RUN python3 -m pip install \
  nksr -f https://nksr.huangjh.tech/whl/torch-2.0.0+cu117.html

RUN python3 -m pip install \
  python-pycg[full] -f https://pycg.huangjh.tech/packages/index.html


RUN apt-get clean

ENV NVIDIA_VISIBLE_DEVICES="all" \
  NVIDIA_DRIVER_CAPABILITIES="video,compute,utility,graphics" \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib:/usr/lib:/usr/local/lib \
  QT_GRAPHICSSYSTEM="native"
