FROM ubuntu:24.04

ENV PIP_BREAK_SYSTEM_PACKAGES 1

# set working directory
WORKDIR /app

# install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    wget \
    gegl \
    unzip \
    libgl1 \
    libglx-mesa0 \
    git-lfs \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN git clone https://github.com/comfyanonymous/ComfyUI
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install -r ComfyUI/requirements.txt
RUN pip3 install fastapi[standard]

WORKDIR /app/ComfyUI/custom_nodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes
RUN pip install -r ComfyUI-KJNodes/requirements.txt
RUN git clone https://github.com/cubiq/ComfyUI_essentials
RUN pip install -r ComfyUI_essentials/requirements.txt

# download models
WORKDIR /app

RUN wget https://huggingface.co/Kijai/PrecompiledWheels/resolve/main/triton-3.3.0-cp312-cp312-linux_x86_64.whl
RUN pip3 install triton-3.3.0-cp312-cp312-linux_x86_64.whl

RUN wget https://huggingface.co/Kijai/PrecompiledWheels/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
RUN pip3 install sageattention-2.2.0-cp312-cp312-linux_x86_64.whl

RUN wget https://huggingface.co/Comfy-Org/OneReward_repackaged/resolve/main/split_files/diffusion_models/flux.1-fill-dev-OneReward-transformer_fp8.safetensors -O ComfyUI/models/diffusion_models/flux.1-fill-dev-OneReward-transformer_fp8.safetensors

RUN wget https://huggingface.co/camenduru/FLUX.1-dev/resolve/fc63f3204a12362f98c04bc4c981a06eb9123eee/FLUX.1-Turbo-Alpha.safetensors -O ComfyUI/models/loras/FLUX.1-Turbo-Alpha.safetensors


#RUN wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors -O ComfyUI/models/clip/t5xxl_fp8_e4m3fn_scaled.safetensors
RUN wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors -O ComfyUI/models/clip/t5xxl_fp16.safetensors
RUN wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors -O ComfyUI/models/clip/clip_l.safetensors
RUN wget https://huggingface.co/fofr/comfyui/resolve/main/vae/ae.safetensors -O ComfyUI/models/vae/ae.safetensors


# copy source files
COPY . .

# run fastapi app
CMD python3 genfill_api.py