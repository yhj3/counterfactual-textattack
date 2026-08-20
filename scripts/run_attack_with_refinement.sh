#!/bin/bash
# Example: CUDA_VISIBLE_DEVICES=2 ./scripts/run_attack_with_refinement.sh
#
# HF_TOKEN must be set — Llama-2 is a gated model on the Hugging Face Hub.

set -euo pipefail

export PYTHONPATH="$PWD/src"

python src/run_attack.py
