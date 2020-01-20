#!/bin/bash

# before start - from the main dir run:
# docker build -t allrank:latest .

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"

docker run -e PYTHONPATH=/allrank -v $PROJECT_DIR:/allrank allrank:latest /bin/sh -c 'python allrank/data/generate_dummy_data.py && python allrank/main.py --config-file-name /allrank/scripts/local_config.json --run-id test_run --job-dir /allrank/task-data'