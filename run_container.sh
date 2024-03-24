#! /bin/bash

docker run \
  --gpus all \
  -it \
  -v .:/app \
  ngc_with_transformer_engine:pytorch-24.02
