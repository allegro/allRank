#!/usr/bin/env bash
set -eu

# before start - from the main dir run:
# docker build -t allrank:latest .

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"

docker run -v $PROJECT_DIR:/allrank allrank:latest /bin/sh -c 'make lint tests'