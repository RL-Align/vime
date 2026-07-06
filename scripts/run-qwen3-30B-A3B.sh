#!/bin/bash

WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
VIME_PYTHON_ENV=${VIME_PYTHON_ENV:-${WORKSPACE_ROOT}/vime-rlk-env}
if [[ -d "${VIME_PYTHON_ENV}/bin" ]]; then
   export PATH="${VIME_PYTHON_ENV}/bin:${PATH}"
fi

# for rerun the task
if [[ "${VIME_SKIP_PROCESS_CLEANUP:-0}" != "1" ]]; then
   pkill -9 -f "vllm serve"
   sleep 3
   ray stop --force
   pkill -9 ray
   pkill -9 python
   sleep 3
   pkill -9 ray
   pkill -9 python
   pkill -9 redis
fi

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
if [[ -z "${CUDA_HOME:-}" ]]; then
   if [[ -d "${VIME_PYTHON_ENV}/lib/python3.11/site-packages/nvidia/cu13" ]]; then
      export CUDA_HOME="${VIME_PYTHON_ENV}/lib/python3.11/site-packages/nvidia/cu13"
   elif [[ -d /usr/local/lib/python3.11/dist-packages/nvidia/cu13 ]]; then
      export CUDA_HOME=/usr/local/lib/python3.11/dist-packages/nvidia/cu13
   else
      export CUDA_HOME=/usr/local/cuda
   fi
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
if [[ -d "${VIME_PYTHON_ENV}/lib/python3.11/site-packages/nvidia/cudnn" ]]; then
   CUDNN_HOME="${VIME_PYTHON_ENV}/lib/python3.11/site-packages/nvidia/cudnn"
else
   CUDNN_HOME="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn"
fi
TORCH_LIB_DIR="${VIME_PYTHON_ENV}/lib/python3.11/site-packages/torch/lib"
if [[ -d "${TORCH_LIB_DIR}" ]]; then
   export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDNN_HOME}/lib:${LD_LIBRARY_PATH:-}"
else
   export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDNN_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi
export CPATH="${CUDA_HOME}/include:${CUDNN_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDNN_HOME}/lib:${LIBRARY_PATH:-}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "NCCL_NVLS_ENABLE: $NCCL_NVLS_ENABLE"
echo "NCCL_CUMEM_ENABLE: $NCCL_CUMEM_ENABLE"

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    DETECTED_GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
else
    DETECTED_GPUS=0
    DETECTED_GPU_NAME="unknown"
fi

validate_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
        echo "${name} must be a positive integer, got '${value}'" >&2
        exit 1
    fi
}

validate_at_most_num_gpus() {
    local name="$1"
    local value="$2"
    if [ "$value" -gt "$NUM_GPUS" ]; then
        echo "${name}=${value} cannot exceed NUM_GPUS=${NUM_GPUS}" >&2
        exit 1
    fi
}

validate_divides_num_gpus() {
    local name="$1"
    local value="$2"
    if [ $((NUM_GPUS % value)) -ne 0 ]; then
        echo "${name}=${value} must divide NUM_GPUS=${NUM_GPUS}" >&2
        exit 1
    fi
}

NUM_GPUS=${NUM_GPUS:-2}
validate_positive_int "NUM_GPUS" "$NUM_GPUS"
if [ "$DETECTED_GPUS" -gt 0 ] && [ "$NUM_GPUS" -gt "$DETECTED_GPUS" ]; then
    echo "Requested NUM_GPUS=$NUM_GPUS but only detected $DETECTED_GPUS GPUs" >&2
    exit 1
fi
echo "BENCHMARK_GPU: ${DETECTED_GPU_NAME}"
echo "NUM_GPUS: $NUM_GPUS"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
MEGATRON_ROOT=${MEGATRON_ROOT:-${WORKSPACE_ROOT}/Megatron-LM}
QWEN3_30B_A3B_HF_DIR=${QWEN3_30B_A3B_HF_DIR:-${WORKSPACE_ROOT}/Qwen3-30B-A3B}
QWEN3_30B_A3B_TORCH_DIST_DIR=${QWEN3_30B_A3B_TORCH_DIST_DIR:-${WORKSPACE_ROOT}/Qwen3-30B-A3B_torch_dist}
DAPO_MATH_17K_DIR=${DAPO_MATH_17K_DIR:-${WORKSPACE_ROOT}/dapo-math-17k}
AIME_2024_DIR=${AIME_2024_DIR:-${WORKSPACE_ROOT}/aime-2024}
VIME_NO_MOE_PERMUTE_FUSION=${VIME_NO_MOE_PERMUTE_FUSION:-1}
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

MEGATRON_TP=${MEGATRON_TP:-2}
MEGATRON_EP=${MEGATRON_EP:-${NUM_GPUS}}
MEGATRON_CP=${MEGATRON_CP:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
NUM_ROLLOUT=${NUM_ROLLOUT:-24}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-1024}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-${NUM_GPUS}}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.5}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-}
VIME_CKPT_DIR=${VIME_CKPT_DIR:-${WORKSPACE_ROOT}/Qwen3-30B-A3B_vime_tp2_dev}
VIME_DISABLE_SAVE=${VIME_DISABLE_SAVE:-1}
VIME_SKIP_EVAL_BEFORE_TRAIN=${VIME_SKIP_EVAL_BEFORE_TRAIN:-1}
VIME_VLLM_ENFORCE_EAGER=${VIME_VLLM_ENFORCE_EAGER:-1}
VIME_LOAD_DEBUG_ROLLOUT_DATA=${VIME_LOAD_DEBUG_ROLLOUT_DATA:-}
VIME_NO_GRAD_ACCUM_FUSION=${VIME_NO_GRAD_ACCUM_FUSION:-1}
VIME_NO_MASKED_SOFTMAX_FUSION=${VIME_NO_MASKED_SOFTMAX_FUSION:-1}
VIME_TRANSFORMER_IMPL=${VIME_TRANSFORMER_IMPL:-local}
VIME_NO_ROPE_FUSION=${VIME_NO_ROPE_FUSION:-1}
VIME_NO_PERSIST_LAYER_NORM=${VIME_NO_PERSIST_LAYER_NORM:-1}
VIME_SEQUENCE_PARALLEL=${VIME_SEQUENCE_PARALLEL:-0}
MEGATRON_ALLOW_MOE_TP_WITHOUT_SP=${MEGATRON_ALLOW_MOE_TP_WITHOUT_SP:-0}
VIME_USE_DISTRIBUTED_OPTIMIZER=${VIME_USE_DISTRIBUTED_OPTIMIZER:-1}
VIME_USE_PRECISION_AWARE_OPTIMIZER=${VIME_USE_PRECISION_AWARE_OPTIMIZER:-1}
VIME_OPTIMIZER_CPU_OFFLOAD=${VIME_OPTIMIZER_CPU_OFFLOAD:-1}
VIME_USE_FP32_GRAD_BUFFER=${VIME_USE_FP32_GRAD_BUFFER:-1}
VIME_GRAD_REDUCE_IN_BF16=${VIME_GRAD_REDUCE_IN_BF16:-0}
VIME_TRAIN_MEMORY_MARGIN_BYTES=${VIME_TRAIN_MEMORY_MARGIN_BYTES:-1073741824}
VIME_DDP_BUCKET_SIZE=${VIME_DDP_BUCKET_SIZE:-}
VIME_DDP_NUM_BUCKETS=${VIME_DDP_NUM_BUCKETS:-}
VIME_ONLY_TRAIN_PARAMS_NAME_LIST=${VIME_ONLY_TRAIN_PARAMS_NAME_LIST:-}
VIME_SYNC_TRAINABLE_WEIGHTS_ONLY=${VIME_SYNC_TRAINABLE_WEIGHTS_ONLY:-0}
VIME_USE_KL_LOSS=${VIME_USE_KL_LOSS:-1}
VIME_USE_ROLLOUT_LOGPROBS=${VIME_USE_ROLLOUT_LOGPROBS:-0}
if [[ "${VIME_RL_KERNEL:-0}" == "1" ]]; then
   VIME_RL_KERNEL_LINEAR_LOGP_BACKEND=${VIME_RL_KERNEL_LINEAR_LOGP_BACKEND:-cuda}
   VIME_RL_KERNEL_CUDA_EVENT_TIMER=${VIME_RL_KERNEL_CUDA_EVENT_TIMER:-1}
   RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16=${RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16:-1}
   if [[ -z "${VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN:-}" && "${VIME_ONLY_TRAIN_PARAMS_NAME_LIST}" == *"output_layer"* ]]; then
      VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN=1
   fi
fi

validate_positive_int "MEGATRON_TP" "$MEGATRON_TP"
validate_positive_int "MEGATRON_EP" "$MEGATRON_EP"
validate_positive_int "MEGATRON_CP" "$MEGATRON_CP"
validate_positive_int "MAX_TOKENS_PER_GPU" "$MAX_TOKENS_PER_GPU"
validate_positive_int "NUM_ROLLOUT" "$NUM_ROLLOUT"
validate_positive_int "ROLLOUT_BATCH_SIZE" "$ROLLOUT_BATCH_SIZE"
validate_positive_int "N_SAMPLES_PER_PROMPT" "$N_SAMPLES_PER_PROMPT"
validate_positive_int "ROLLOUT_MAX_RESPONSE_LEN" "$ROLLOUT_MAX_RESPONSE_LEN"
validate_positive_int "GLOBAL_BATCH_SIZE" "$GLOBAL_BATCH_SIZE"
validate_positive_int "ROLLOUT_NUM_GPUS_PER_ENGINE" "$ROLLOUT_NUM_GPUS_PER_ENGINE"
if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
   validate_positive_int "VLLM_MAX_MODEL_LEN" "$VLLM_MAX_MODEL_LEN"
fi
if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
   validate_positive_int "VLLM_MAX_NUM_SEQS" "$VLLM_MAX_NUM_SEQS"
fi
validate_at_most_num_gpus "MEGATRON_TP" "$MEGATRON_TP"
validate_at_most_num_gpus "MEGATRON_EP" "$MEGATRON_EP"
validate_at_most_num_gpus "ROLLOUT_NUM_GPUS_PER_ENGINE" "$ROLLOUT_NUM_GPUS_PER_ENGINE"
validate_divides_num_gpus "MEGATRON_TP" "$MEGATRON_TP"
validate_divides_num_gpus "MEGATRON_EP" "$MEGATRON_EP"
validate_divides_num_gpus "ROLLOUT_NUM_GPUS_PER_ENGINE" "$ROLLOUT_NUM_GPUS_PER_ENGINE"

echo "MEGATRON_TP: $MEGATRON_TP"
echo "MEGATRON_EP: $MEGATRON_EP"
echo "MEGATRON_CP: $MEGATRON_CP"
echo "ROLLOUT_NUM_GPUS_PER_ENGINE: $ROLLOUT_NUM_GPUS_PER_ENGINE"
echo "NUM_ROLLOUT: $NUM_ROLLOUT"
echo "ROLLOUT_BATCH_SIZE: $ROLLOUT_BATCH_SIZE"
echo "N_SAMPLES_PER_PROMPT: $N_SAMPLES_PER_PROMPT"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "MAX_TOKENS_PER_GPU: $MAX_TOKENS_PER_GPU"
echo "ROLLOUT_MAX_RESPONSE_LEN: $ROLLOUT_MAX_RESPONSE_LEN"
echo "VLLM_GPU_MEMORY_UTILIZATION: $VLLM_GPU_MEMORY_UTILIZATION"
echo "VLLM_MAX_MODEL_LEN: ${VLLM_MAX_MODEL_LEN:-<unset>}"
echo "VLLM_MAX_NUM_SEQS: ${VLLM_MAX_NUM_SEQS:-<unset>}"
echo "WORKSPACE_ROOT: $WORKSPACE_ROOT"
echo "MEGATRON_ROOT: $MEGATRON_ROOT"
echo "QWEN3_30B_A3B_HF_DIR: $QWEN3_30B_A3B_HF_DIR"
echo "QWEN3_30B_A3B_TORCH_DIST_DIR: $QWEN3_30B_A3B_TORCH_DIST_DIR"
echo "DAPO_MATH_17K_DIR: $DAPO_MATH_17K_DIR"
echo "AIME_2024_DIR: $AIME_2024_DIR"
echo "VIME_CKPT_DIR: $VIME_CKPT_DIR"
echo "VIME_LOAD_DEBUG_ROLLOUT_DATA: ${VIME_LOAD_DEBUG_ROLLOUT_DATA:-<unset>}"
echo "VIME_USE_DISTRIBUTED_OPTIMIZER: $VIME_USE_DISTRIBUTED_OPTIMIZER"
echo "VIME_USE_PRECISION_AWARE_OPTIMIZER: $VIME_USE_PRECISION_AWARE_OPTIMIZER"
echo "VIME_OPTIMIZER_CPU_OFFLOAD: $VIME_OPTIMIZER_CPU_OFFLOAD"
echo "VIME_USE_FP32_GRAD_BUFFER: $VIME_USE_FP32_GRAD_BUFFER"
echo "VIME_GRAD_REDUCE_IN_BF16: $VIME_GRAD_REDUCE_IN_BF16"
echo "VIME_TRAIN_MEMORY_MARGIN_BYTES: $VIME_TRAIN_MEMORY_MARGIN_BYTES"
echo "VIME_TRANSFORMER_IMPL: $VIME_TRANSFORMER_IMPL"
echo "VIME_ONLY_TRAIN_PARAMS_NAME_LIST: ${VIME_ONLY_TRAIN_PARAMS_NAME_LIST:-<unset>}"
echo "VIME_SYNC_TRAINABLE_WEIGHTS_ONLY: $VIME_SYNC_TRAINABLE_WEIGHTS_ONLY"
echo "VIME_USE_KL_LOSS: $VIME_USE_KL_LOSS"
echo "VIME_USE_ROLLOUT_LOGPROBS: $VIME_USE_ROLLOUT_LOGPROBS"
echo "VIME_RL_KERNEL_LINEAR_LOGP_BACKEND: ${VIME_RL_KERNEL_LINEAR_LOGP_BACKEND:-<unset>}"
echo "VIME_RL_KERNEL_CUDA_EVENT_TIMER: ${VIME_RL_KERNEL_CUDA_EVENT_TIMER:-<unset>}"
echo "VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN: ${VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN:-<auto>}"
echo "RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16: ${RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16:-<unset>}"

CKPT_ARGS=(
   --hf-checkpoint "${QWEN3_30B_A3B_HF_DIR}"
   #--hf-checkpoint /root/Qwen3-30B-A3B-FP8
   --ref-load "${QWEN3_30B_A3B_TORCH_DIST_DIR}"
   --load "${VIME_CKPT_DIR}/"
)
if [[ "${VIME_DISABLE_SAVE:-0}" != "1" ]]; then
   CKPT_ARGS+=(
      --save "${VIME_CKPT_DIR}/"
      --save-interval "${VIME_SAVE_INTERVAL:-20}"
   )
fi

ROLLOUT_ARGS=(
   --prompt-data "${DAPO_MATH_17K_DIR}/dapo-math-17k.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 1

   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "${AIME_2024_DIR}/aime-2024.jsonl"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)
if [[ "${VIME_SKIP_EVAL_BEFORE_TRAIN:-0}" == "1" ]]; then
   EVAL_ARGS+=(--skip-eval-before-train)
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${MEGATRON_TP}"
   --pipeline-model-parallel-size 1
   --context-parallel-size "${MEGATRON_CP}"
   --expert-model-parallel-size "${MEGATRON_EP}"
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if [[ "${VIME_SEQUENCE_PARALLEL:-0}" == "1" ]]; then
   PERF_ARGS+=(--sequence-parallel)
fi
if [[ "${VIME_NO_GRAD_ACCUM_FUSION:-0}" == "1" ]]; then
   PERF_ARGS+=(--no-gradient-accumulation-fusion)
fi
if [[ "${VIME_NO_MASKED_SOFTMAX_FUSION:-0}" == "1" ]]; then
   PERF_ARGS+=(--no-masked-softmax-fusion)
fi
PERF_ARGS+=(--transformer-impl "${VIME_TRANSFORMER_IMPL}")
if [[ "${VIME_NO_ROPE_FUSION:-0}" == "1" ]]; then
   PERF_ARGS+=(--no-rope-fusion)
fi
if [[ "${VIME_NO_PERSIST_LAYER_NORM:-0}" == "1" ]]; then
   PERF_ARGS+=(--no-persist-layer-norm)
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
if [[ "${VIME_USE_KL_LOSS:-1}" == "1" ]]; then
   GRPO_ARGS+=(--use-kl-loss)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
if [[ "${VIME_OPTIMIZER_CPU_OFFLOAD:-1}" == "1" ]]; then
   OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
fi
if [[ "${VIME_USE_PRECISION_AWARE_OPTIMIZER:-1}" == "1" ]]; then
   OPTIMIZER_ARGS+=(--use-precision-aware-optimizer)
fi

WANDB_ARGS=(
   #--use-wandb
   # --wandb-project vime-dev
   # --wandb-group qwen3-30B-A3B-test
   # --wandb-key ${WANDB_KEY}
)

TB_ARGS=()
if [[ "${VIME_TENSORBOARD:-0}" == "1" ]]; then
   export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${VIME_ROOT}/tensorboard_log/${TB_EXPERIMENT_NAME:-qwen3-30B-A3B}}"
   TB_ARGS+=(--use-tensorboard)
   TB_ARGS+=(--tb-project-name "${TB_PROJECT_NAME:-vime-rlk}")
   TB_ARGS+=(--tb-experiment-name "${TB_EXPERIMENT_NAME:-qwen3-30B-A3B}")
fi

VLLM_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
   --vllm-enable-expert-parallel
)
if [[ -n "${VLLM_MAX_MODEL_LEN}" ]]; then
   VLLM_ARGS+=(--vllm-max-model-len "${VLLM_MAX_MODEL_LEN}")
fi
if [[ -n "${VLLM_MAX_NUM_SEQS}" ]]; then
   VLLM_ARGS+=(--vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}")
fi
if [[ "${VIME_VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
   VLLM_ARGS+=(--vllm-enforce-eager)
else
   VLLM_ARGS+=(--vllm-cudagraph-capture-sizes 1 2 4 8 $(seq 16 8 256))
fi

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --train-memory-margin-bytes "${VIME_TRAIN_MEMORY_MARGIN_BYTES}"
   # need to comment this when using model with MLA
   --attention-backend flash
)
if [[ "${VIME_USE_FP32_GRAD_BUFFER:-1}" == "1" ]]; then
   MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi
if [[ "${VIME_GRAD_REDUCE_IN_BF16:-0}" == "1" ]]; then
   MISC_ARGS+=(--grad-reduce-in-bf16)
fi
if [[ "${VIME_USE_ROLLOUT_LOGPROBS:-0}" == "1" ]]; then
   MISC_ARGS+=(--use-rollout-logprobs)
fi
if [[ -n "${VIME_DDP_BUCKET_SIZE}" ]]; then
   MISC_ARGS+=(--ddp-bucket-size "${VIME_DDP_BUCKET_SIZE}")
fi
if [[ -n "${VIME_DDP_NUM_BUCKETS}" ]]; then
   MISC_ARGS+=(--ddp-num-buckets "${VIME_DDP_NUM_BUCKETS}")
fi
if [[ -n "${VIME_LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
   MISC_ARGS+=(--load-debug-rollout-data "${VIME_LOAD_DEBUG_ROLLOUT_DATA}")
fi
if [[ -n "${VIME_ONLY_TRAIN_PARAMS_NAME_LIST}" ]]; then
   IFS=',' read -ra _ONLY_TRAIN_PATTERNS <<< "${VIME_ONLY_TRAIN_PARAMS_NAME_LIST}"
   MISC_ARGS+=(--only-train-params-name-list)
   for _pattern in "${_ONLY_TRAIN_PATTERNS[@]}"; do
      if [[ -n "${_pattern}" ]]; then
         MISC_ARGS+=("${_pattern}")
      fi
   done
fi

RLK_ARGS=()
if [[ "${VIME_RL_KERNEL:-0}" == "1" ]]; then
   RLK_ARGS+=(--enable-rl-kernel --rl-kernel-ops "${VIME_RL_KERNEL_OPS:-linear_logp}")
   if [[ "${VIME_RL_KERNEL_STRICT:-0}" == "1" ]]; then
      RLK_ARGS+=(--rl-kernel-strict)
   fi
fi

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
cd "${VIME_ROOT}"
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${VIME_ROOT}:${MEGATRON_ROOT}\",
    \"PATH\": \"${PATH}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CPATH\": \"${CPATH:-}\",
    \"LIBRARY_PATH\": \"${LIBRARY_PATH:-}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"CUDA_MODULE_LOADING\": \"${CUDA_MODULE_LOADING}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${NCCL_NVLS_ENABLE}\",
    \"NCCL_CUMEM_ENABLE\": \"${NCCL_CUMEM_ENABLE}\",
    \"VIME_USE_DISTRIBUTED_OPTIMIZER\": \"${VIME_USE_DISTRIBUTED_OPTIMIZER}\",
    \"VIME_CPU_MOE_CKPT_MERGE\": \"${VIME_CPU_MOE_CKPT_MERGE:-1}\",
    \"VIME_SYNC_TRAINABLE_WEIGHTS_ONLY\": \"${VIME_SYNC_TRAINABLE_WEIGHTS_ONLY}\",
    \"VIME_RL_KERNEL_LINEAR_LOGP_BACKEND\": \"${VIME_RL_KERNEL_LINEAR_LOGP_BACKEND:-}\",
    \"VIME_RL_KERNEL_CUDA_EVENT_TIMER\": \"${VIME_RL_KERNEL_CUDA_EVENT_TIMER:-0}\",
    \"VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN\": \"${VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN:-}\",
    \"VIME_RL_KERNEL_VALIDATE_TP_TARGETS\": \"${VIME_RL_KERNEL_VALIDATE_TP_TARGETS:-0}\",
    \"RL_KERNEL_LINEAR_LOGP_FUSED_BACKWARD\": \"${RL_KERNEL_LINEAR_LOGP_FUSED_BACKWARD:-1}\",
    \"RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16\": \"${RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16:-0}\",
    \"RL_KERNEL_LINEAR_LOGP_VALIDATE_TP_TARGETS\": \"${RL_KERNEL_LINEAR_LOGP_VALIDATE_TP_TARGETS:-0}\",
    \"VIME_LINEAR_LOGP_MEMORY_PROBE\": \"${VIME_LINEAR_LOGP_MEMORY_PROBE:-0}\",
    \"VIME_BASELINE_LINEAR_LOGP_TIMER\": \"${VIME_BASELINE_LINEAR_LOGP_TIMER:-0}\",
    \"VIME_BASELINE_CUDA_EVENT_TIMER\": \"${VIME_BASELINE_CUDA_EVENT_TIMER:-0}\",
    \"MEGATRON_LOCAL_ATTENTION_SINGLE_PACKED_SEQ\": \"${MEGATRON_LOCAL_ATTENTION_SINGLE_PACKED_SEQ:-0}\",
    \"MEGATRON_ALLOW_MOE_TP_WITHOUT_SP\": \"${MEGATRON_ALLOW_MOE_TP_WITHOUT_SP}\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${TB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${RLK_ARGS[@]}
