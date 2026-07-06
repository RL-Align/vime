#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh CONFIG MODE

CONFIG: T01 | T02 | T03 | T04 | T05 | T06 | T07 | T08
MODE:   baseline | cuda

Important environment overrides:
  TRAIN_SCOPE=full|output_layer      default: full
  TRACE_MODE=none|train|rollout|all  default: all
  TRACE_ROLLOUTS=3                   nsys capture rollout ids
  RUN_ROOT=/workspace                log/trace root
  NSYS=/path/to/nsys                 nsys binary
EOF
}

CONFIG_ID="${1:-}"
MODE="${2:-}"
if [[ -z "${CONFIG_ID}" || -z "${MODE}" ]]; then
  usage
  exit 2
fi
if [[ "${MODE}" != "baseline" && "${MODE}" != "cuda" ]]; then
  usage
  exit 2
fi

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
VENV="${VIME_PYTHON_ENV:-${WORKSPACE_ROOT}/vime-rlk-env}"
NSYS="${NSYS:-/opt/nvidia/nsight-compute/2024.1.1/host/target-linux-x64/nsys}"

case "${CONFIG_ID}" in
  T01)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.40}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
    ;;
  T02)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-3072}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.42}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-3072}"
    ;;
  T03)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1536}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
    ;;
  T04)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
    ;;
  T05)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-6144}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-3072}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
    ;;
  T06)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-3072}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
    ;;
  T07)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-3968}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.46}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
    ;;
  T08)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
    N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
    MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
    ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-3584}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
    ;;
  *)
    usage
    exit 2
    ;;
esac

TRAIN_SCOPE="${TRAIN_SCOPE:-full}"
if [[ "${TRAIN_SCOPE}" != "full" && "${TRAIN_SCOPE}" != "output_layer" ]]; then
  echo "TRAIN_SCOPE must be full or output_layer, got ${TRAIN_SCOPE}" >&2
  exit 2
fi

TRACE_MODE="${TRACE_MODE:-all}"
case "${TRACE_MODE}" in
  none|train|rollout|all) ;;
  *)
    echo "TRACE_MODE must be none, train, rollout, or all; got ${TRACE_MODE}" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-8gpu_${CONFIG_ID}_${TRAIN_SCOPE}_${MODE}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs/${RUN_NAME}}"
TRACE_DIR="${TRACE_DIR:-${RUN_ROOT}/nsys_traces/${RUN_NAME}}"
mkdir -p "${LOG_DIR}" "${TRACE_DIR}"

export NUM_GPUS="${NUM_GPUS:-8}"
export MEGATRON_TP="${MEGATRON_TP:-2}"
export MEGATRON_EP="${MEGATRON_EP:-8}"
export MEGATRON_CP="${MEGATRON_CP:-1}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-12}"
export VIME_DISABLE_SAVE="${VIME_DISABLE_SAVE:-1}"
export VIME_SKIP_EVAL_BEFORE_TRAIN="${VIME_SKIP_EVAL_BEFORE_TRAIN:-1}"
export VIME_USE_KL_LOSS="${VIME_USE_KL_LOSS:-0}"
export VIME_SKIP_ZERO_ENTROPY_METRIC="${VIME_SKIP_ZERO_ENTROPY_METRIC:-1}"
export VIME_VLLM_ENFORCE_EAGER="${VIME_VLLM_ENFORCE_EAGER:-1}"
export VIME_RL_KERNEL_OPS="${VIME_RL_KERNEL_OPS:-linear_logp}"
export VIME_RL_KERNEL_LINEAR_LOGP_BACKEND="${VIME_RL_KERNEL_LINEAR_LOGP_BACKEND:-cuda}"
export VIME_RL_KERNEL_CUDA_EVENT_TIMER="${VIME_RL_KERNEL_CUDA_EVENT_TIMER:-1}"
export VIME_LINEAR_LOGP_MEMORY_PROBE="${VIME_LINEAR_LOGP_MEMORY_PROBE:-1}"
export VIME_TRAIN_MEMORY_MARGIN_BYTES="${VIME_TRAIN_MEMORY_MARGIN_BYTES:-1073741824}"
export MEGATRON_LOCAL_ATTENTION_SINGLE_PACKED_SEQ="${MEGATRON_LOCAL_ATTENTION_SINGLE_PACKED_SEQ:-1}"
export VIME_USE_DISTRIBUTED_OPTIMIZER="${VIME_USE_DISTRIBUTED_OPTIMIZER:-1}"
export VIME_USE_PRECISION_AWARE_OPTIMIZER="${VIME_USE_PRECISION_AWARE_OPTIMIZER:-1}"
export VIME_OPTIMIZER_CPU_OFFLOAD="${VIME_OPTIMIZER_CPU_OFFLOAD:-1}"
export VIME_USE_FP32_GRAD_BUFFER="${VIME_USE_FP32_GRAD_BUFFER:-0}"
export VIME_GRAD_REDUCE_IN_BF16="${VIME_GRAD_REDUCE_IN_BF16:-1}"
export VIME_SEQUENCE_PARALLEL="${VIME_SEQUENCE_PARALLEL:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"
export VIME_SKIP_PROCESS_CLEANUP="${VIME_SKIP_PROCESS_CLEANUP:-0}"

export ROLLOUT_BATCH_SIZE
export N_SAMPLES_PER_PROMPT
export GLOBAL_BATCH_SIZE
export MAX_TOKENS_PER_GPU
export ROLLOUT_MAX_RESPONSE_LEN
export VLLM_GPU_MEMORY_UTILIZATION
export VLLM_MAX_MODEL_LEN

if [[ "${MODE}" == "cuda" ]]; then
  export VIME_RL_KERNEL=1
  export VIME_BASELINE_LINEAR_LOGP_TIMER=0
  export VIME_BASELINE_CUDA_EVENT_TIMER=0
  if [[ "${TRAIN_SCOPE}" == "output_layer" ]]; then
    export VIME_ONLY_TRAIN_PARAMS_NAME_LIST="${VIME_ONLY_TRAIN_PARAMS_NAME_LIST:-output_layer}"
    export VIME_SYNC_TRAINABLE_WEIGHTS_ONLY="${VIME_SYNC_TRAINABLE_WEIGHTS_ONLY:-1}"
    export VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN="${VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN:-1}"
    export RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16="${RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16:-1}"
    export RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL="${RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL:-0}"
  else
    export VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN="${VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN:-0}"
    export RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16="${RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16:-0}"
    export RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL="${RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL:-1}"
  fi
else
  export VIME_RL_KERNEL=0
  export VIME_BASELINE_LINEAR_LOGP_TIMER=1
  export VIME_BASELINE_CUDA_EVENT_TIMER=1
  export RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16=0
  export RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL=0
  if [[ "${TRAIN_SCOPE}" == "output_layer" ]]; then
    export VIME_ONLY_TRAIN_PARAMS_NAME_LIST="${VIME_ONLY_TRAIN_PARAMS_NAME_LIST:-output_layer}"
    export VIME_SYNC_TRAINABLE_WEIGHTS_ONLY="${VIME_SYNC_TRAINABLE_WEIGHTS_ONLY:-1}"
    export VIME_BASELINE_OUTPUT_LAYER_DETACH_HIDDEN="${VIME_BASELINE_OUTPUT_LAYER_DETACH_HIDDEN:-1}"
  else
    export VIME_BASELINE_OUTPUT_LAYER_DETACH_HIDDEN="${VIME_BASELINE_OUTPUT_LAYER_DETACH_HIDDEN:-0}"
  fi
fi

case "${TRACE_MODE}" in
  none)
    export VIME_NSYS_CAPTURE_ROLLOUTS=""
    export VIME_NSYS_CAPTURE_ROLE="${VIME_NSYS_CAPTURE_ROLE:-actor}"
    ;;
  train)
    export VIME_NSYS_CAPTURE_ROLLOUTS="${TRACE_ROLLOUTS:-${VIME_NSYS_CAPTURE_ROLLOUTS:-3}}"
    export VIME_NSYS_CAPTURE_ROLE="actor"
    ;;
  rollout)
    export VIME_NSYS_CAPTURE_ROLLOUTS="${TRACE_ROLLOUTS:-${VIME_NSYS_CAPTURE_ROLLOUTS:-3}}"
    export VIME_NSYS_CAPTURE_ROLE="rollout"
    ;;
  all)
    export VIME_NSYS_CAPTURE_ROLLOUTS="${TRACE_ROLLOUTS:-${VIME_NSYS_CAPTURE_ROLLOUTS:-3}}"
    export VIME_NSYS_CAPTURE_ROLE="all"
    ;;
esac

if [[ "${TRACE_MODE}" == "rollout" || "${TRACE_MODE}" == "all" ]]; then
  if [[ -z "${VLLM_PROFILER_CONFIG:-}" ]]; then
    export VLLM_PROFILER_CONFIG='{"profiler":"cuda"}'
  else
    export VLLM_PROFILER_CONFIG
  fi
fi

TOTAL_SAMPLES_PER_ROLLOUT=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
TRAIN_DP_SIZE=$((NUM_GPUS / MEGATRON_TP / MEGATRON_CP))
if (( GLOBAL_BATCH_SIZE > TOTAL_SAMPLES_PER_ROLLOUT )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} exceeds ROLLOUT_BATCH_SIZE*N_SAMPLES_PER_PROMPT=${TOTAL_SAMPLES_PER_ROLLOUT}" >&2
  exit 2
fi

if (( TOTAL_SAMPLES_PER_ROLLOUT < TRAIN_DP_SIZE )); then
  echo "RBS*NSP=${TOTAL_SAMPLES_PER_ROLLOUT} is smaller than training DP size ${TRAIN_DP_SIZE}; each DP rank needs at least one sample" >&2
  exit 2
fi

export CONFIG_ID MODE TRAIN_SCOPE TRACE_MODE RUN_NAME LOG_DIR TRACE_DIR

echo "run_name=${RUN_NAME}" | tee "${LOG_DIR}/run_env.log"
env | sort | grep -E '^(CONFIG_ID|MODE|TRAIN_SCOPE|TRACE_MODE|NUM_GPUS|MEGATRON_|ROLLOUT_|N_SAMPLES|GLOBAL_BATCH|MAX_TOKENS|VLLM_|VIME_|RL_KERNEL_|NCCL_|CUDA_MODULE_LOADING|LOG_DIR|TRACE_DIR|RUN_NAME)=' \
  | tee -a "${LOG_DIR}/run_env.log"

run_train() {
  bash "${VIME_ROOT}/scripts/run-qwen3-30B-A3B.sh"
}

NSYS_RAY_PID=""
cleanup() {
  set +e
  if [[ -n "${NSYS_RAY_PID}" ]] && kill -0 "${NSYS_RAY_PID}" >/dev/null 2>&1; then
    "${VENV}/bin/ray" stop --force >/dev/null 2>&1 || true
    for _ in $(seq 1 120); do
      if ! kill -0 "${NSYS_RAY_PID}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if kill -0 "${NSYS_RAY_PID}" >/dev/null 2>&1; then
      kill -TERM "${NSYS_RAY_PID}" >/dev/null 2>&1 || true
    fi
    wait "${NSYS_RAY_PID}" >/dev/null 2>&1 || true
  fi
  pkill -TERM -f "[v]llm serve" >/dev/null 2>&1 || true
  pkill -TERM -f "[V]LLM::" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "${VIME_ROOT}"
if [[ "${TRACE_MODE}" == "none" ]]; then
  run_train 2>&1 | tee "${LOG_DIR}/ray_job_${MODE}.log"
else
  if [[ ! -x "${NSYS}" ]]; then
    echo "nsys not found or not executable: ${NSYS}" >&2
    exit 1
  fi

  "${VENV}/bin/ray" stop --force >/dev/null 2>&1 || true
  pkill -9 -f "[v]llm serve" >/dev/null 2>&1 || true
  pkill -9 -f "[V]LLM::" >/dev/null 2>&1 || true
  sleep 2

  "${NSYS}" profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none \
    --cpuctxsw=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat:64 \
    --flush-on-cudaprofilerstop=true \
    --wait=all \
    --force-overwrite=true \
    --export=sqlite \
    --output="${TRACE_DIR}/${RUN_NAME}" \
    "${VENV}/bin/ray" start \
      --head \
      --node-ip-address 127.0.0.1 \
      --num-gpus "${NUM_GPUS}" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8265 \
      --block \
    >"${LOG_DIR}/nsys_ray_start.log" 2>&1 &
  NSYS_RAY_PID=$!

  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:8265/api/version" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${NSYS_RAY_PID}" >/dev/null 2>&1; then
      echo "nsys/ray process exited early; see ${LOG_DIR}/nsys_ray_start.log" >&2
      exit 1
    fi
    sleep 1
  done
  if ! curl -fsS "http://127.0.0.1:8265/api/version" >/dev/null 2>&1; then
    echo "ray dashboard did not become ready; see ${LOG_DIR}/nsys_ray_start.log" >&2
    exit 1
  fi

  VIME_SKIP_PROCESS_CLEANUP=1 VIME_SKIP_RAY_START=1 run_train 2>&1 | tee "${LOG_DIR}/ray_job_${MODE}.log"
fi

cleanup
trap - EXIT

echo "trace files:"
find "${TRACE_DIR}" -maxdepth 1 -type f -print | sort || true
echo "logs:"
find "${LOG_DIR}" -maxdepth 1 -type f -print | sort || true
