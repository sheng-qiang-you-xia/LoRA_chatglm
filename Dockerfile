#指定版本的cuda和pytorch
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
#设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
#安装python3.8和其他必要系统工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
#安装pytorch 1.10.0 和cuda 11.3 版本的兼容 \
RUN pip3 install torch==11.3+cu113 \
    torchvision==0.11.1+cu113 \
    torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html \
#安装 其他的库 \
RUN pip3 install \
    transformers==4.27.1 \
    protobuf>=3.19.5,<3.20.1 \
    icetk \
    cpm_kernels \
    streamlit==1.17.0 \
    matplotlib \
    datasets==2.10.1 \
    accelerate==0.17.0 \
    packaging>=20.0 \
    psutil \
    pyyaml \
    peft \
    deepspeed==0.15.4 \
    triton \
    flash-attn --no-build-isolation \
    nvitop \


