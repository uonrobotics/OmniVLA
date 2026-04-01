#!/usr/bin/env bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate omnivla

python /home/sujin/workspace/physical-ai/OmniVLA/inference/finetune_model/omnivla_client.py