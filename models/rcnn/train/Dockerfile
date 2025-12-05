#docker build -t fasterrcnn:python310 .
#docker run -it --gpus all -v .:/app -v ./../../../storage:/storage --name fasterrcnn fasterrcnn:python310 bash

FROM python:3.10

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y locales locales-all
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

ARG BOT_STORAGE_ROOT=/storage
ENV BOT_MODELS_ROOT=$BOT_STORAGE_ROOT/models
ENV BOT_DATA_ROOT=$BOT_STORAGE_ROOT/data
ENV ENVIRONMENT=local

ENV TORCH_HOME=$BOT_STORAGE_ROOT/data/cached_models/torch
ENV HF_HOME=$BOT_STORAGE_ROOT/data/cached_models/hf
ENV HUGGINGFACE_HUB_CACHE=$BOT_STORAGE_ROOT/data/cached_models/hf/hub
ENV TRANSFORMERS_CACHE=$BOT_STORAGE_ROOT/data/cached_models/hf/transformers
ENV SENTENCE_TRANSFORMERS_HOME=$BOT_STORAGE_ROOT/data/cached_models/sentence-transformers
ENV TIMM_HOME=$BOT_STORAGE_ROOT/data/cached_models/timm

RUN apt update
RUN apt install -y --no-install-recommends python3-pip python3-dev nano wget git
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install --no-cache-dir setuptools
RUN pip install --upgrade pip

WORKDIR /app
ADD . /app

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install pillow tqdm torchmetrics pycocotools