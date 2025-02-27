FROM continuumio/miniconda3

RUN \
 conda update -n base -c defaults conda -y --quiet && \
 conda clean --all
RUN apt-get update && apt-get install -y python3-opencv && apt-get install python3.9-dev -y \
 && apt-get install build-essential -y

COPY conda_env.yaml /conda_env.yaml

SHELL ["/bin/bash", "-c"]

RUN \
 conda env create --file conda_env.yaml -vv

RUN conda clean --all && rm /conda_env.yaml

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.3
