#!/usr/bin/env bash
# Qwen3-8B GRPO smoke/validation run with the RL-Kernel linear_logp provider.
# Vime remains the launcher; RL-Kernel owns the provider and its
# contract.  This script intentionally keeps the framework-side change small.

set -euo pipefail

VIME_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RL_KERNEL_ROOT="${RL_KERNEL_ROOT:-${VIME_ROOT}/../RL-Kernel}"
RL_KERNEL_MODE="${RL_KERNEL_MODE:-strict}"
RL_KERNEL_ALIGNED="${RL_KERNEL_ALIGNED:-1}"

case "${RL_KERNEL_MODE}" in
  strict|audit)
    RL_KERNEL_CASE=R/R
    LINEAR_LOGP_PROVIDER_MODE=strict
    ;;
  auto)
    RL_KERNEL_CASE=P/P
    LINEAR_LOGP_PROVIDER_MODE=auto
    ;;
  off)
    RL_KERNEL_CASE=P/P
    LINEAR_LOGP_PROVIDER_MODE=
    ;;
  *)
    echo "RL_KERNEL_MODE must be strict, audit, auto, or off" >&2
    exit 2
    ;;
esac
if [[ "${RL_KERNEL_ALIGNED}" != "0" && "${RL_KERNEL_ALIGNED}" != "1" ]]; then
  echo "RL_KERNEL_ALIGNED must be 0 or 1" >&2
  exit 2
fi

export RL_KERNEL_MODE
if [[ "${RL_KERNEL_MODE}" != off ]]; then
  if [[ ! -f "${RL_KERNEL_ROOT}/rl_engine/integrations/vime/linear_logp_provider.py" ]]; then
    echo "RL_KERNEL_ROOT must contain the RL-Kernel Vime linear_logp provider" >&2
    exit 2
  fi
  export RL_KERNEL_ATTENTION_CASE="${RL_KERNEL_CASE}"
  export RL_KERNEL_FFN_CASE="${RL_KERNEL_CASE}"
  export RL_KERNEL_LOGP_CASE="${RL_KERNEL_CASE}"
  export RL_KERNEL_VLLM_INTEGRATION=1
  export RL_KERNEL_CUDA_ONLY=1
  export VIME_RL_KERNEL_STRICT=0
  if [[ "${RL_KERNEL_MODE}" == strict || "${RL_KERNEL_MODE}" == audit ]]; then
    export VIME_RL_KERNEL_STRICT=1
  fi
  if [[ "${RL_KERNEL_MODE}" == audit ]]; then
    export RL_KERNEL_ROUTE_REPORT_ALL_RANKS=1
  fi
else
  unset RL_KERNEL_VLLM_INTEGRATION VIME_RL_KERNEL_STRICT
  unset RL_KERNEL_DET_GEMM_SM90_ONLY RL_KERNEL_CUDA_ONLY
  unset RL_KERNEL_ATTENTION_CASE RL_KERNEL_FFN_CASE RL_KERNEL_LOGP_CASE
fi

export PYTHONUNBUFFERED=1
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
if [[ "${RL_KERNEL_MODE}" == off ]]; then
  export PYTHONPATH="${VIME_ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${RL_KERNEL_ROOT}:${VIME_ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"
fi
if [[ "${RL_KERNEL_ALIGNED}" == 1 ]]; then
  unset RL_KERNEL_DET_GEMM_SM90_ONLY
  export VLLM_BATCH_INVARIANT=1
  export NCCL_ALGO=Ring
  export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
  export CUBLAS_WORKSPACE_CONFIG=:16:8
  export CUBLASLT_WORKSPACE_SIZE=1
elif [[ "${RL_KERNEL_MODE}" != off ]]; then
  export RL_KERNEL_DET_GEMM_SM90_ONLY=1
fi

# The provider is a vocab-parallel TP implementation.  CP owns token rows and
# must not be used as a vocabulary reduction group.
TP_SIZE="${TP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-2}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
COLOCATE="${COLOCATE:-0}"

if [[ "${NUM_GPUS}" != "8" ]]; then
  echo "This validation entry point requires an 8-GPU node" >&2
  exit 2
fi
if [[ "${COLOCATE}" != "0" && "${COLOCATE}" != "1" ]]; then
  echo "COLOCATE must be 0 (default, disjoint train/rollout GPUs) or 1" >&2
  exit 2
fi

if [[ "$((TP_SIZE * CP_SIZE))" != "${ACTOR_GPUS}" ]]; then
  echo "TP_SIZE * CP_SIZE must equal ACTOR_GPUS" >&2
  exit 2
fi
if [[ "${COLOCATE}" == "1" ]]; then
  if [[ "${ACTOR_GPUS}" != "8" || "${ROLLOUT_GPUS}" != "8" ]]; then
    echo "Colocated execution requires 8 actor GPUs and 8 logical rollout GPUs" >&2
    exit 2
  fi
elif [[ "${ACTOR_GPUS}" != "4" || "${ROLLOUT_GPUS}" != "4" ]]; then
  echo "Disjoint execution requires 4 actor GPUs and 4 rollout GPUs" >&2
  exit 2
fi
if [[ "${ROLLOUT_GPUS_PER_ENGINE}" != "${TP_SIZE}" ]]; then
  echo "ROLLOUT_GPUS_PER_ENGINE must equal TP_SIZE for aligned logp validation" >&2
  exit 2
fi
if (( ROLLOUT_GPUS % ROLLOUT_GPUS_PER_ENGINE != 0 )); then
  echo "ROLLOUT_GPUS must be divisible by ROLLOUT_GPUS_PER_ENGINE" >&2
  exit 2
fi
if [[ "${RL_KERNEL_MODE}" != off \
  && "${RL_KERNEL_MODE}" != auto \
  && "${ROLLOUT_TOP_P}" != "1.0" ]]; then
  echo "RL-Kernel strict linear_logp validation requires ROLLOUT_TOP_P=1.0" >&2
  exit 2
fi

source "${VIME_ROOT}/scripts/models/qwen3-8B.sh"

MODEL_ROOT="${MODEL_ROOT:-/root/Qwen3-8B}"
TORCH_DIST_ROOT="${TORCH_DIST_ROOT:-/root/Qwen3-8B_torch_dist}"
VIME_CKPT="${VIME_CKPT:-/root/Qwen3-8B_vime_rlkernel_tp${TP_SIZE}_cp${CP_SIZE}}"
PROMPT_DATA="${PROMPT_DATA:-/root/dapo-math-17k/dapo-math-17k.jsonl}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required; refusing to run the CUDA validation on an unknown device" >&2
  exit 3
fi
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
GPU_COUNT="$(printf '%s\n' "${GPU_NAMES}" | sed '/^$/d' | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" != "${NUM_GPUS}" ]]; then
  echo "Expected ${NUM_GPUS} visible GPUs, found ${GPU_COUNT}" >&2
  printf '%s\n' "${GPU_NAMES}" >&2
  exit 3
fi
if [[ "${GPU_REQUIRE_H100:-1}" == "1" ]] && ! printf '%s\n' "${GPU_NAMES}" | grep -q 'H100'; then
  echo "Expected H100 GPUs; refusing to run on a different GPU class" >&2
  printf '%s\n' "${GPU_NAMES}" >&2
  exit 3
fi
python3 - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit("PyTorch must expose 8 CUDA devices for this validation")
PY
for required_path in "${MODEL_ROOT}" "${TORCH_DIST_ROOT}" "${PROMPT_DATA}" "${MEGATRON_ROOT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required runtime path does not exist: ${required_path}" >&2
    exit 3
  fi
done
if [[ "${RL_KERNEL_MODE}" != off ]]; then
  python3 - <<'PY'
from rl_engine.integrations.vime.linear_logp_provider import provider
print(f"RL-Kernel provider import OK: {provider.__module__}.{provider.__name__}")
PY
fi

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

RL_KERNEL_ARGS=()
if [[ "${RL_KERNEL_MODE}" != off ]]; then
  RL_KERNEL_ARGS=(
    --linear-logp-provider rl_engine.integrations.vime.linear_logp_provider.provider
    --linear-logp-provider-mode "${LINEAR_LOGP_PROVIDER_MODE}"
    --custom-megatron-init-path rl_engine.integrations.megatron_runtime.initialize_from_environment
  )
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend flash
  --no-gradient-accumulation-fusion
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
)
if [[ "${RL_KERNEL_ALIGNED}" == 1 ]]; then
  MISC_ARGS+=(
    --seed 1234
    --rollout-seed 42
    --vllm-enable-deterministic-inference
    --vllm-attention-backend flash_attn
    --vllm-disable-custom-all-reduce
    --deterministic-mode
    --accumulate-allreduce-grads-in-fp32
  )
fi

ray stop --force || true
ray start --head --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \
  --num-gpus "${NUM_GPUS}" --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT:-8265}"

TRAIN_LAYOUT_ARGS=()
if [[ "${COLOCATE}" == "1" ]]; then
  TRAIN_LAYOUT_ARGS+=(--colocate)
else
  TRAIN_LAYOUT_ARGS+=(--megatron-to-hf-mode bridge)
fi

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-8265}" \
  --working-dir "${VIME_ROOT}" \
  -- python3 train.py \
  --train-backend megatron \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  "${TRAIN_LAYOUT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${PARALLEL_ARGS[@]}" \
  "${RL_KERNEL_ARGS[@]}" \
  "${MISC_ARGS[@]}"
