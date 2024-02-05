#!/usr/bin/env bash

# Copyright 2024 StarfleetAI
# SPDX-License-Identifier: Apache-2.0

set -e

echo "Training 'Open-Orca/Mistral-7B-OpenOrca' on 'StarfleetAI/polaris-dataset' to 'out'"
accelerate launch \
    --config_file ./fsdp_config.yaml \
    train.py \
    --checkpoint Open-Orca/Mistral-7B-OpenOrca \
    --dataset ./polaris-dataset \
    --output_dir out \
    --wandb_project "polaris-small" \
    --num_epochs 2

echo "Done!"
echo "You can find the trained model in 'out' directory"
