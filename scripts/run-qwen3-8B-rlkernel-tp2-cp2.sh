#!/usr/bin/env bash
# Qwen3-8B GRPO smoke/validation run with the RL-Kernel selected-logprob
# provider.  Vime remains the launcher; RL-Kernel owns the provider and its
# contract.  This script intentionally keeps the framework-side change small.

set -euo pipefail

VIME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RL_KERNEL_ROOT="${RL_KERNEL_ROOT:-${VIME_ROOT}/../RL-Kernel}"

if [[ ! -f "${RL_KERNEL_ROOT}/rl_engine/integrations/vime/logp.py" ]]; then
  echo "RL_KERNEL_ROOT must point to an RL-Kernel checkout containing the Vime provider" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${RL_KERNEL_ROOT}:${VIME_ROOT}:/root/Megatron-LM:${PYTHONPATH:-}"

# The provider is a vocab-parallel TP implementation.  CP owns token rows and
# must not be used as a vocabulary reduction group.
TP_SIZE="${TP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-2}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"

if [[ "${TP_SIZE}" != "2" || "${CP_SIZE}" != "2" ]]; then
  echo "This validation entry point is intentionally fixed to TP=2, CP=2" >&2
  exit 2
fi
if [[ "${ROLLOUT_TOP_P}" != "1.0" ]]; then
  echo "RL-Kernel strict selected-logprob validation requires ROLLOUT_TOP_P=1.0" >&2
  exit 2
fi

source "${VIME_ROOT}/scripts/models/qwen3-8B.sh"

MODEL_ROOT="${MODEL_ROOT:-/root/Qwen3-8B}"
TORCH_DIST_ROOT="${TORCH_DIST_ROOT:-/root/Qwen3-8B_torch_dist}"
VIME_CKPT="${VIME_CKPT:-/root/Qwen3-8B_vime_rlkernel_tp2_cp2}"
PROMPT_DATA="${PROMPT_DATA:-/root/dapo-math-17k/dapo-math-17k.jsonl}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_ROOT}"
  --ref-load "${TORCH_DIST_ROOT}"
  --load "${VIME_CKPT}"
  --save "${VIME_CKPT}"
  --save-interval 100000
)

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type deepscaler
  --num-rollout "${NUM_ROLLOUT:-1}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-2}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN:-1024}"
  --rollout-temperature 1.0
  --rollout-top-p "${ROLLOUT_TOP_P}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
  --balance-data
)

PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --pipeline-model-parallel-size 1
  --sequence-parallel
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-2048}"
)

RL_KERNEL_ARGS=(
  --selected-logprob-provider rl_engine.integrations.vime.logp.provider
  --selected-logprob-provider-mode strict
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend flash
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
)

ray stop --force || true
ray start --head --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \
  --num-gpus "${ACTOR_GPUS}" --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT:-8265}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}" \
  --working-dir "${VIME_ROOT}" \
  -- python3 train.py \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${PARALLEL_ARGS[@]}" \
  "${RL_KERNEL_ARGS[@]}" \
  "${MISC_ARGS[@]}"
