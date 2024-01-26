#!/usr/bin/env bash

# Copyright 2024 StarfleetAI
# SPDX-License-Identifier: Apache-2.0

set -e

echo "Training `Open-Orca/Mistral-7B-OpenOrca` on `ajibawa-2023/Code-290k-ShareGPT` to `tmp`"
accelerate launch \
    --config_file ./fsdp_config.yaml \
    train.py \
    --checkpoint Open-Orca/Mistral-7B-OpenOrca \
    --dataset ajibawa-2023/Code-290k-ShareGPT \
    --output_dir tmp \
    --num_epochs 1

echo "Training `tmp` on `StarfleetAI/function-calling` to `out`"
accelerate launch \
    --config_file ./fsdp_config.yaml \
    train.py \
    --checkpoint tmp \
    --dataset StarfleetAI/function-calling \
    --output_dir out \
    --num_epochs 3

echo "Removing `tmp`"
rm -rf tmp

echo "Done!"
echo "You can find the trained model in `out` directory"
