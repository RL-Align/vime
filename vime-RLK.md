# vime + RL-Kernel linear_logp 2xH100 最小开发验证

## 0. 我们要做什么

本轮不是主宣传 benchmark，而是 2xH100 最小开发验证：

```text
candidate: RL-Align/vime#2 + RL-Align/RL-Kernel#189
model:     Qwen3-30B-A3B
hardware:  2xH100 colocate
op:        RL-Kernel linear_logp
```

目标只验证三件事：

- vime 可以在 2xH100 上启动 Qwen3-30B-A3B 最小训练链路。
- `VIME_RL_KERNEL=1` 后能进入 RL-Kernel `linear_logp` 路径。
- `VIME_RL_KERNEL_STRICT=1` 下 `rl_kernel_fallback_count = 0`，至少完成 1 个 train step。

不产出宣传结论；不比较速度收益；不画最终 benchmark 图。

## 1. 范围

只保留：

- `linear_logp`
- Qwen3-30B-A3B
- TP=2
- 2xH100 单机 colocate
- candidate 必跑，baseline 只做可选环境 sanity check

不做：

- 8xH100 主宣传实验
- Qwen3-4B smoke
- R3 单独对比
- GLM-4.5
- GB200/H200/A100 硬件对照
- 训推一致性专项 benchmark
- MoE expert/router RL-Kernel 算子

## 2. 最小配置

从极小 batch 开始，先保证代码路径跑通：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
export MEGATRON_TP=2
export MEGATRON_EP=2
export MEGATRON_CP=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=2

export NUM_ROLLOUT=8
export ROLLOUT_BATCH_SIZE=1
export N_SAMPLES_PER_PROMPT=1
export GLOBAL_BATCH_SIZE=1
export MAX_TOKENS_PER_GPU=2048
export ROLLOUT_MAX_RESPONSE_LEN=512
export VLLM_GPU_MEMORY_UTILIZATION=0.45

export VIME_CKPT_DIR=/root/Qwen3-30B-A3B_vime_tp2_dev
export VIME_DISABLE_SAVE=1
export VIME_SKIP_EVAL_BEFORE_TRAIN=1
export VIME_VLLM_ENFORCE_EAGER=1
export VIME_NO_GRAD_ACCUM_FUSION=1
```

如果这组能跑通，再逐步放大：

```text
MAX_TOKENS_PER_GPU=4096
ROLLOUT_MAX_RESPONSE_LEN=1024
ROLLOUT_BATCH_SIZE=2
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=4
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

## 6. 跑 candidate

```bash
cd /workspace/vime-rlk-tp2

export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
export MEGATRON_TP=2
export MEGATRON_EP=2
export MEGATRON_CP=1
export ROLLOUT_NUM_GPUS_PER_ENGINE=2

export NUM_ROLLOUT=8
export ROLLOUT_BATCH_SIZE=1
export N_SAMPLES_PER_PROMPT=1
export GLOBAL_BATCH_SIZE=1
export MAX_TOKENS_PER_GPU=2048
export ROLLOUT_MAX_RESPONSE_LEN=512
export VLLM_GPU_MEMORY_UTILIZATION=0.45

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

## 7. 可选 baseline sanity check

baseline 只用于确认环境和 vime 脚本本身能跑，不用于性能对比。

```bash
cd /workspace/vime-rlk-tp2
unset VIME_RL_KERNEL VIME_RL_KERNEL_OPS VIME_RL_KERNEL_STRICT
bash scripts/run-qwen3-30B-A3B.sh 2>&1 | tee /workspace/vime-rlk-tp2-baseline.log
```

## 8. 验收线

candidate 日志必须满足：

```text
RL-Kernel linear_logp backend 被加载
VIME_RL_KERNEL_STRICT=1 没有触发 RuntimeError
rl_kernel_fallback_count = 0
至少完成 1 个 train step
log_probs / loss / reward 指标为 finite
```

允许：

```text
step time 不稳定
reward 无明显趋势
吞吐很低
显存接近上限
```

不允许：

```text
fallback 到 vime materialized logits 路径
target vocab shard 报错
TP collective hang
loss/logprob NaN 或 Inf
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
first_successful_train_step
peak_vram_gb
error_stack_if_failed
```

## 10. 下一步

2xH100 通过后再进入正式 benchmark：

```text
8xH100
Qwen3-30B-A3B
baseline vs candidate
至少 3 次 run
统计 step time、logprob time、peak VRAM、raw_reward、train_rollout_logprob_abs_diff
```

只有 8xH100 正式 benchmark 结果可以进入宣传材料。
