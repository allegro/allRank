#!/usr/bin/env bash

DIR=$(dirname $0)
PROJECT_DIR="$(cd $DIR/..; pwd)"

docker build -t allrank:latest $PROJECT_DIR
$PROJECT_DIR/scripts/run_tests.sh