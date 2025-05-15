FROM wowai/base-hf:v1.12.0
WORKDIR /app

ENV MODEL_DIR=/data/models
ENV RQ_QUEUE_NAME=default
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV PORT=9090
ENV AIXBLOCK_USE_REDIS=false
ENV HOST_NAME=https://dev-us-west-1.aixblock.io
ENV HF_TOKEN=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

ENV PYTHONUNBUFFERED=True \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache


RUN --mount=type=cache,target=/root/.cache 

# COPY uwsgi.ini /etc/uwsgi
RUN apt-get -qq update && \
   DEBIAN_FRONTEND=noninteractive \ 
   apt-get install --no-install-recommends --assume-yes git

RUN apt install build-essential
RUN apt install -y ffmpeg libpq-dev uwsgi libpq-dev python3-dev
RUN apt install -y nvidia-cuda-toolkit --fix-missing
RUN apt-get -qq -y install curl --fix-missing

RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu12.2

COPY requirements.txt .
RUN python3.10 -m pip install -r requirements.txt

COPY . ./

#GPU
# RUN python3.10 -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu oneccl_bind_pt==2.5.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

#CPU
# RUN python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# RUN python3.10 -m pip install intel-extension-for-pytorch oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/cn/

# RUN git clone https://huggingface.co/tonyshark/deepseek-v3-1b . && python3.10 -m pip  install --no-cache-dir -r /app/inference/requirements.txt
# RUN python3.10  /app/inference/convert.py --hf-ckpt-path deepdeek-v3-1b --save-path /app/data/checkpoint --n-experts 256 --model-parallel 2

EXPOSE 9090 6006 12345 23456
CMD exec gunicorn --preload --bind :${PORT} --workers 1 --threads 1 --timeout 0 _wsgi:app
