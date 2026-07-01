# vime + RL-Kernel linear_logp 2xH100 性能预验证

## 0. 我们要做什么

本轮先在 2xH100 上做 A/B 性能预验证，不做 smoke-only。只有 2 卡已经明显优于 vime 原生路径，才扩大到 8 卡主宣传 benchmark。

```text
baseline:  RL-Align/vime#2, RL-Kernel off, Qwen3-30B-A3B, 2xH100 colocate
candidate: RL-Align/vime#2 + RL-Align/RL-Kernel#189, RL-Kernel linear_logp on, Qwen3-30B-A3B, 2xH100 colocate
```

2 卡阶段仍然使用和 8 卡一致的指标验收线：

- `rl_kernel_fallback_count = 0`
- `raw_reward` 不低于 baseline 同量级
- `train_rollout_logprob_abs_diff` 不持续高于 baseline
- `mean_log_probs_time_s` 或 `peak_vram_gb` 有明确下降
- 最好能看到明显收益后再上 8 卡：建议 `mean_log_probs_time_s` 下降 >= 20% 或 `peak_vram_gb` 下降 >= 10%

2 卡结果只作为上 8 卡前的门禁，不直接进入宣传材料；但该门禁必须放大 selected-logprob workload，能看出 RL-Kernel `linear_logp` 的真实收益。

## 1. 范围

只保留：

- `linear_logp`
- Qwen3-30B-A3B
- TP=2
- 2xH100 单机 colocate
- baseline 和 candidate 都必须跑
- 指标集合与 8xH100 主 benchmark 保持一致

不做：

- 8xH100 主宣传实验
- Qwen3-4B smoke
- R3 单独对比
- GLM-4.5
- GB200/H200/A100 硬件对照
- 训推一致性专项 benchmark
- MoE expert/router RL-Kernel 算子

## 2. 性能预验证配置

默认配置不是 smoke，而是 24 step 的 2 卡性能预验证。核心思路是增加 selected-logprob token 数，让 `linear_logp` 的收益不要被 rollout、update weights 等固定开销完全淹没。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
export MEGATRON_TP=2
export MEGATRON_EP=2
export MEGATRON_CP=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=2

export NUM_ROLLOUT=24
export ROLLOUT_BATCH_SIZE=2
export N_SAMPLES_PER_PROMPT=2
export GLOBAL_BATCH_SIZE=4
export MAX_TOKENS_PER_GPU=4096
export ROLLOUT_MAX_RESPONSE_LEN=1024
export VLLM_GPU_MEMORY_UTILIZATION=0.50

export VIME_CKPT_DIR=/root/Qwen3-30B-A3B_vime_tp2_dev
export VIME_DISABLE_SAVE=1
export VIME_SKIP_EVAL_BEFORE_TRAIN=1
export VIME_VLLM_ENFORCE_EAGER=1
export VIME_NO_GRAD_ACCUM_FUSION=1
```

如果 2xH100 OOM，先只做这一档降级；降级后仍然不是 smoke，因为 response len 和 step 数保持较大：

```text
MAX_TOKENS_PER_GPU=4096
ROLLOUT_MAX_RESPONSE_LEN=1024
ROLLOUT_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=2
```

## 3. 拉代码

从官方仓库开始：

```bash
cd /workspace
git clone https://github.com/RL-Align/RL-Kernel.git RL-Kernel
git clone https://github.com/RL-Align/vime.git vime-rlk-tp2
```

RL-Kernel 使用 TP 版 `linear_logp`：

```bash
cd /workspace/RL-Kernel
git checkout main
git pull origin main
gh pr checkout 189
```

vime 使用 2xH100 开发验证 PR：

```bash
cd /workspace/vime-rlk-tp2
git checkout main
git pull origin main
gh pr checkout 2
```

## 4. 安装

```bash
cd /workspace/RL-Kernel
pip install -e .
python setup.py build_ext --inplace -v

cd /workspace/vime-rlk-tp2
pip install -e .
```

## 5. 模型和数据

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login

hf download Qwen/Qwen3-30B-A3B --local-dir /root/Qwen3-30B-A3B

hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

转换 Megatron `torch_dist` checkpoint：

```bash
cd /workspace/vime-rlk-tp2
source scripts/models/qwen3-30B-A3B.sh

PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 2 \
  tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen3-30B-A3B \
  --save /root/Qwen3-30B-A3B_torch_dist

mkdir -p /root/Qwen3-30B-A3B_vime_tp2_dev
```

## 6. 跑 baseline

baseline 必跑，用来代表 vime 原生路径；不要打开 RL-Kernel。

```bash
cd /workspace/vime-rlk-tp2

export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
export MEGATRON_TP=2
export MEGATRON_EP=2
export MEGATRON_CP=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=2

export NUM_ROLLOUT=24
export ROLLOUT_BATCH_SIZE=2
export N_SAMPLES_PER_PROMPT=2
export GLOBAL_BATCH_SIZE=4
export MAX_TOKENS_PER_GPU=4096
export ROLLOUT_MAX_RESPONSE_LEN=1024
export VLLM_GPU_MEMORY_UTILIZATION=0.50

export VIME_CKPT_DIR=/root/Qwen3-30B-A3B_vime_tp2_dev
export VIME_DISABLE_SAVE=1
export VIME_SKIP_EVAL_BEFORE_TRAIN=1
export VIME_VLLM_ENFORCE_EAGER=1
export VIME_NO_GRAD_ACCUM_FUSION=1

unset VIME_RL_KERNEL VIME_RL_KERNEL_OPS VIME_RL_KERNEL_STRICT

bash scripts/run-qwen3-30B-A3B.sh 2>&1 | tee /workspace/vime-rlk-tp2-baseline.log
```

## 7. 跑 candidate

candidate 使用同一套 2 卡配置，只打开 RL-Kernel。

```bash
cd /workspace/vime-rlk-tp2

export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
export MEGATRON_TP=2
export MEGATRON_EP=2
export MEGATRON_CP=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=2

export NUM_ROLLOUT=24
export ROLLOUT_BATCH_SIZE=2
export N_SAMPLES_PER_PROMPT=2
export GLOBAL_BATCH_SIZE=4
export MAX_TOKENS_PER_GPU=4096
export ROLLOUT_MAX_RESPONSE_LEN=1024
export VLLM_GPU_MEMORY_UTILIZATION=0.50

export VIME_CKPT_DIR=/root/Qwen3-30B-A3B_vime_tp2_dev
export VIME_DISABLE_SAVE=1
export VIME_SKIP_EVAL_BEFORE_TRAIN=1
export VIME_VLLM_ENFORCE_EAGER=1
export VIME_NO_GRAD_ACCUM_FUSION=1

export VIME_RL_KERNEL=1
export VIME_RL_KERNEL_OPS=linear_logp
export VIME_RL_KERNEL_STRICT=1

bash scripts/run-qwen3-30B-A3B.sh 2>&1 | tee /workspace/vime-rlk-tp2-candidate.log
```

## 8. 验收线

每组先跑 1 次确认无错误；稳定后 baseline/candidate 各跑至少 3 次，丢弃前 5-10 step warmup 后统计。

candidate 必须满足：

```text
RL-Kernel linear_logp backend 被加载
VIME_RL_KERNEL_STRICT=1 没有触发 RuntimeError
rl_kernel_fallback_count = 0
rl_kernel_linear_logp_call_count_delta > 0
rl_kernel_linear_logp_token_count_delta > 0
rl_kernel_linear_logp_dispatch_elapsed_s_delta > 0
log_probs / loss / reward 指标为 finite
raw_reward 不低于 baseline 同量级
train_rollout_logprob_abs_diff 不持续高于 baseline
mean_log_probs_time_s 或 peak_vram_gb 有明确下降
```

2 卡上卡门槛：

```text
每组至少 24 train step
丢弃前 5 step warmup
mean_log_probs_time_s 下降 >= 20%
或 peak_vram_gb 下降 >= 10%
或二者都有小幅但稳定下降，且 mean_step_time_s 不明显变差
```

不允许：

```text
fallback 到 vime materialized logits 路径
target vocab shard 报错
TP collective hang
loss/logprob NaN 或 Inf
candidate 质量指标明显劣于 baseline
rl_kernel_linear_logp_call_count_delta 长时间为 0
rl_kernel_linear_logp_token_count_delta 只覆盖极少 token
```

runtime counter 解释：

```text
*_total：当前进程累计命中的 RL-Kernel linear_logp 调用、token 和 dispatch 耗时。
*_delta：两次 train log 之间新增的调用、token 和 dispatch 耗时；第一个 train step 会覆盖此前 ref-logprob 加本 step train-logprob。
tokens_per_call = token_count_delta / max(call_count_delta, 1)，用于判断是否只是空调用或很小 workload。
dispatch_elapsed_s 不做 CUDA synchronize，不作为 GPU kernel time 宣传；正式性能仍看 mean_log_probs_time_s、step time 和 profiler。
```

## 9. 必须记录

```text
gpu_name
num_gpus
vime_commit
rl_kernel_commit
vime_pr
rl_kernel_pr
model
tp
ep
cp
rollout_batch_size
n_samples_per_prompt
global_batch_size
max_tokens_per_gpu
rollout_max_response_len
vllm_gpu_memory_utilization
selected_rl_kernel_backend
rl_kernel_fallback_count
rl_kernel_linear_logp_call_count_total
rl_kernel_linear_logp_call_count_delta
rl_kernel_linear_logp_token_count_total
rl_kernel_linear_logp_token_count_delta
rl_kernel_linear_logp_dispatch_elapsed_s_total
rl_kernel_linear_logp_dispatch_elapsed_s_delta
rl_kernel_linear_logp_tokens_per_call_total
rl_kernel_linear_logp_tokens_per_call_delta
first_successful_train_step
mean_step_time_s
p50_step_time_s
p90_step_time_s
mean_log_probs_time_s
p50_log_probs_time_s
p90_log_probs_time_s
peak_vram_gb
raw_reward_mean
train_rollout_logprob_abs_diff_mean
error_stack_if_failed
```

## 10. 下一步

2xH100 指标门禁通过后再进入正式 benchmark：

```text
8xH100
Qwen3-30B-A3B
baseline vs candidate
至少 3 次 run
统计 step time、logprob time、peak VRAM、raw_reward、train_rollout_logprob_abs_diff
```

只有 8xH100 正式 benchmark 结果可以进入宣传材料。
