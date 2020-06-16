#!/bin/bash

# before start - from the main dir run:
# docker build -t allrank:latest .

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"

docker run -e PYTHONPATH=/allrank -v $PROJECT_DIR:/allrank allrank:latest /bin/sh -c 'python allrank/rank_and_click.py --config-file-name /allrank/scripts/local_config_click_model.json --input-model-path /allrank/task-data/results/test_run/model.pkl --run-id test_run_click --roles train,vali --job-dir /allrank/task-data'