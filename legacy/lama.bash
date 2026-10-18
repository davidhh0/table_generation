#!/bin/bash

NGPUS=4
CHECKPOINT_DIR=~/.llama/checkpoints/Llama-4-Scout-17B-16E-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --nproc_per_node=$NGPUS \
  -m models.llama4.scripts.chat_completion $CHECKPOINT_DIR \
  --world_size $NGPUS