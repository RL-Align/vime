#!/usr/bin/env bash
# Qwen3-8B TP4/CP2 GRPO consistency run on one colocated 8-GPU node.

set -euo pipefail

export TP_SIZE=4
export CP_SIZE=2
export ACTOR_GPUS=8
export ROLLOUT_GPUS=8
export NUM_GPUS=8
export ROLLOUT_GPUS_PER_ENGINE=4
export COLOCATE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run-qwen3-8B-rlkernel-tp2-cp2.sh" "$@"
