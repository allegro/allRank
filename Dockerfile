ARG arch_version

FROM python:3.10 as base

MAINTAINER MLR <allrank@allegro.pl>

RUN mkdir /allrank
COPY requirements.txt setup.py Makefile README.md /allrank/

RUN make -C /allrank install-reqs

WORKDIR /allrank

FROM base as CPU
RUN python3 -m pip  install torchvision==0.14.1 torch==1.13.1  --extra-index-url https://download.pytorch.org/whl/cpu

FROM base as GPU
RUN python3 -m pip  install torchvision==0.14.1 torch==1.13.1  

FROM ${arch_version} as FINAL
