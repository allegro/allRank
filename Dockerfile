FROM python:3.7

MAINTAINER MLR <allrank@allegro.pl>

RUN mkdir /allrank
COPY requirements.txt setup.py Makefile README.md /allrank/

RUN make -C /allrank install-reqs

WORKDIR /allrank