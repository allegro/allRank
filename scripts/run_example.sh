#!/usr/bin/env bash

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"

docker build -t allrank:latest $PROJECT_DIR
docker run -e PYTHONPATH=/allrank -v $PROJECT_DIR:/allrank allrank:latest /bin/sh -c 'python allrank/data/generate_dummy_data.py && python allrank/main.py --config-file-name /allrank/scripts/local_config.json --run-id test_run --job-dir /allrank/test_run'
