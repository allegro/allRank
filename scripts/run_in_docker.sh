#!/bin/bash

# before start - from the main dir run:
# docker build -t allrank:latest .
# need to run python setup.py install before running locally

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"

docker run -e PYTHONPATH=/allrank -v $PROJECT_DIR:/allrank allrank:latest /bin/sh -c 'python allrank/main.py --config-file-name config.json --run-id test_run --output /allrank/task-data'