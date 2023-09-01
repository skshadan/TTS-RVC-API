#!/bin/bash
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker
set -e

DATE=$(date -u +"%Y%m%dT%H%MZ")
docker build -t registry.gitlab.com/alexchadwick/bark-rvc-api:$DATE .
docker tag registry.gitlab.com/alexchadwick/bark-rvc-api:$DATE registry.gitlab.com/alexchadwick/bark-rvc-api:latest
docker push registry.gitlab.com/alexchadwick/bark-rvc-api:$DATE

docker run \
    --name api \
    --rm \
    --runtime=nvidia \
    --gpus all \
    -p 8000:8000 \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -v ./config.toml:/opt/config.toml \
    -v /mnt/fstore/models/voice:/models \
    -v ~/Downloads:/tmp/ \
    -v ~/.cache/suno:/root/.cache/suno/ \
    -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
    -v ~/.cache/torch:/root/.cache/torch \
    registry.gitlab.com/alexchadwick/bark-rvc-api:$DATE

# docker push registry.gitlab.com/alexchadwick/bark-rvc-api:latest