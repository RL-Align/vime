#!/bin/bash

# for rerun the task
pkill -9 -f "vllm serve"
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python
pkill -9 redis

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    DETECTED_GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
else
    DETECTED_GPUS=0
    DETECTED_GPU_NAME="unknown"
fi
NUM_GPUS=${NUM_GPUS:-8}
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -le 0 ]; then
    NUM_GPUS=8
fi
if [ "$DETECTED_GPUS" -gt 0 ] && [ "$NUM_GPUS" -gt "$DETECTED_GPUS" ]; then
    echo "Requested NUM_GPUS=$NUM_GPUS but only detected $DETECTED_GPUS GPUs" >&2
    exit 1
fi
echo "BENCHMARK_GPU: ${DETECTED_GPU_NAME}"
echo "NUM_GPUS: $NUM_GPUS"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

MEGATRON_TP=${MEGATRON_TP:-2}
MEGATRON_EP=${MEGATRON_EP:-${NUM_GPUS}}
MEGATRON_CP=${MEGATRON_CP:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-20480}
NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-${NUM_GPUS}}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.7}
VIME_CKPT_DIR=${VIME_CKPT_DIR:-/root/Qwen3-30B-A3B_vime}

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-30B-A3B
   #--hf-checkpoint /root/Qwen3-30B-A3B-FP8
   --ref-load /root/Qwen3-30B-A3B_torch_dist
   --load "${VIME_CKPT_DIR}/"
)
if [[ "${VIME_DISABLE_SAVE:-0}" != "1" ]]; then
   CKPT_ARGS+=(
      --save "${VIME_CKPT_DIR}/"
      --save-interval "${VIME_SAVE_INTERVAL:-20}"
   )
fi

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
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
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)
if [[ "${VIME_SKIP_EVAL_BEFORE_TRAIN:-0}" == "1" ]]; then
   EVAL_ARGS+=(--skip-eval-before-train)
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${MEGATRON_TP}"
   --sequence-parallel
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
if [[ "${VIME_NO_GRAD_ACCUM_FUSION:-0}" == "1" ]]; then
   PERF_ARGS+=(--no-gradient-accumulation-fusion)
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

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
if [[ "${VIME_VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
   VLLM_ARGS+=(--vllm-enforce-eager)
else
   VLLM_ARGS+=(--vllm-cudagraph-capture-sizes 1 2 4 8 $(seq 16 8 256))
fi

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

RLK_ARGS=()
if [[ "${VIME_RL_KERNEL:-0}" == "1" ]]; then
   RLK_ARGS+=(--enable-rl-kernel --rl-kernel-ops "${VIME_RL_KERNEL_OPS:-linear_logp}")
   if [[ "${VIME_RL_KERNEL_STRICT:-0}" == "1" ]]; then
      RLK_ARGS+=(--rl-kernel-strict)
   fi
fi

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${VIME_ROOT}:/root/Megatron-LM/\",
    \"PATH\": \"${PATH}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"CPATH\": \"${CPATH:-}\",
    \"LIBRARY_PATH\": \"${LIBRARY_PATH:-}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
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
