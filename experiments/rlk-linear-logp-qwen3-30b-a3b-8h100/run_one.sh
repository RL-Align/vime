#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
   echo "Usage: $0 <baseline|candidate> <run-name> <repo-dir> <run-dir>" >&2
   exit 2
fi

VARIANT="$1"
RUN_NAME="$2"
REPO_DIR="$3"
RUN_DIR="$4"

if [[ "${VARIANT}" != "baseline" && "${VARIANT}" != "candidate" ]]; then
   echo "VARIANT must be baseline or candidate, got ${VARIANT}" >&2
   exit 2
fi

mkdir -p "${RUN_DIR}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/lib/python3.11/dist-packages/nvidia/cu13}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/include:${CPATH:-}"
export LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:${CUDA_HOME}/lib:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MEGATRON_TP="${MEGATRON_TP:-8}"
export MEGATRON_EP="${MEGATRON_EP:-8}"
export MEGATRON_CP="${MEGATRON_CP:-1}"
export ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-20480}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
export VIME_VLLM_ENFORCE_EAGER="${VIME_VLLM_ENFORCE_EAGER:-1}"
export VIME_NO_GRAD_ACCUM_FUSION="${VIME_NO_GRAD_ACCUM_FUSION:-1}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-12}"

export VIME_TENSORBOARD=1
export TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
export TB_PROJECT_NAME="${TB_PROJECT_NAME:-vime-rlk-linear-logp}"
export TB_EXPERIMENT_NAME="${RUN_NAME}"
export VIME_CKPT_DIR="${RUN_DIR}/ckpt"
export VIME_DISABLE_SAVE="${VIME_DISABLE_SAVE:-1}"
export VIME_SAVE_INTERVAL="${VIME_SAVE_INTERVAL:-20}"
export VIME_SKIP_EVAL_BEFORE_TRAIN="${VIME_SKIP_EVAL_BEFORE_TRAIN:-1}"

if [[ "${VARIANT}" == "candidate" ]]; then
   export VIME_RL_KERNEL=1
   export VIME_RL_KERNEL_OPS=linear_logp
   export VIME_RL_KERNEL_STRICT=1
else
   unset VIME_RL_KERNEL VIME_RL_KERNEL_OPS VIME_RL_KERNEL_STRICT
fi

{
   echo "variant=${VARIANT}"
   echo "run_name=${RUN_NAME}"
   echo "repo_dir=${REPO_DIR}"
   echo "run_dir=${RUN_DIR}"
   echo "cuda_home=${CUDA_HOME}"
   echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
   echo "num_gpus=${NUM_GPUS}"
   echo "megatron_tp=${MEGATRON_TP}"
   echo "megatron_ep=${MEGATRON_EP}"
   echo "megatron_cp=${MEGATRON_CP}"
   echo "rollout_num_gpus_per_engine=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
   echo "rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
   echo "n_samples_per_prompt=${N_SAMPLES_PER_PROMPT}"
   echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
   echo "max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}"
   echo "vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION}"
   echo "vime_vllm_enforce_eager=${VIME_VLLM_ENFORCE_EAGER}"
   echo "vime_no_grad_accum_fusion=${VIME_NO_GRAD_ACCUM_FUSION}"
   echo "rollout_max_response_len=${ROLLOUT_MAX_RESPONSE_LEN}"
   echo "num_rollout=${NUM_ROLLOUT}"
   echo "vime_save_interval=${VIME_SAVE_INTERVAL}"
   echo "vime_disable_save=${VIME_DISABLE_SAVE}"
   echo "vime_skip_eval_before_train=${VIME_SKIP_EVAL_BEFORE_TRAIN}"
   echo "vime_rl_kernel=${VIME_RL_KERNEL:-0}"
   echo "vime_rl_kernel_ops=${VIME_RL_KERNEL_OPS:-}"
   echo "vime_rl_kernel_strict=${VIME_RL_KERNEL_STRICT:-0}"
   nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
} > "${RUN_DIR}/run_config.txt"

(
   while true; do
      date +%s
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
      sleep "${VRAM_POLL_INTERVAL:-2}"
   done
) > "${RUN_DIR}/vram.csv" &
MONITOR_PID=$!

cleanup() {
   kill "${MONITOR_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "${REPO_DIR}"
bash scripts/run-qwen3-30B-A3B.sh 2>&1 | tee "${RUN_DIR}/train.log"
