# vime + RL-Kernel linear_logp 8 卡训推 12 轮运行步骤

日期：2026-07-06 UTC

## 0. 运行原则

- 8 卡正式结果必须完整训推 12 轮：`NUM_ROLLOUT=12`。
- 模型固定 Qwen3-30B-A3B。
- 不设置 `LOAD_DEBUG_ROLLOUT_DATA`，不走 debug train-only。
- 先跑 candidate full-gradient 找最大不 OOM，再补 baseline。
- 每个成功配置必须完成 rollout 0-11；取 rollout 3-11 的均值填表。
- trace 用 `TRACE_MODE=train|rollout|all`；默认 `all`，需要少 overhead 时才改成 `train` 或 `none`。

## 1. 准备环境

```bash
cd /workspace/vime-rlk-tp2
git status --short
nvidia-smi -L
```

确认 RL-Kernel 扩展：

```bash
/workspace/vime-rlk-env/bin/python - <<'PY'
import torch
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("rl_kernel_ext", _EXT_AVAILABLE)
print("fused_bwd", hasattr(_C, "fused_linear_logp_sm90_backward"))
PY
```

确认模型和数据：

```bash
test -d /workspace/Qwen3-30B-A3B && echo "HF ok"
test -d /workspace/Qwen3-30B-A3B_torch_dist && echo "torch_dist ok"
test -f /workspace/dapo-math-17k/dapo-math-17k.jsonl && echo "prompt ok"
```

## 2. 运行脚本

统一使用：

```bash
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh CONFIG MODE
```

参数：

| 参数 | 可选值 |
| --- | --- |
| `CONFIG` | `T01`, `T02`, `T03`, `T04`, `T05`, `T06`, `T07`, `T08` |
| `MODE` | `cuda`, `baseline` |
| `TRAIN_SCOPE` | `full`, `output_layer` |
| `TRACE_MODE` | `none`, `train`, `rollout`, `all` |
| `TRACE_ROLLOUTS` | `3`, `3-5`, `all` |

默认：

- `TRAIN_SCOPE=full`
- `TRACE_MODE=all`
- `TRACE_ROLLOUTS=3`
- `NUM_GPUS=8`
- `MEGATRON_TP=2`
- `MEGATRON_EP=8`
- `NUM_ROLLOUT=12`

trace 文件写到：

```text
/workspace/nsys_traces/<run_name>/
```

日志写到：

```text
/workspace/logs/<run_name>/
```

配置：

| 配置 | 参数 |
| --- | --- |
| T01 | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=2048; RESP=512; VLLM_MEM=0.40; VLLM_MAX_MODEL_LEN=2048` |
| T02 | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=3072; RESP=1024; VLLM_MEM=0.42; VLLM_MAX_MODEL_LEN=3072` |
| T03 | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=4096; RESP=1536; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` |
| T04 | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=4096; RESP=2048; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` |
| T05 | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=6144; RESP=3072; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` |
| T06 | `RBS=4; NSP=2; GBS=8; MAX_TOKENS=8192; RESP=3072; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` |
| T07 | `RBS=4; NSP=2; GBS=8; MAX_TOKENS=8192; RESP=3968; VLLM_MEM=0.46; VLLM_MAX_MODEL_LEN=4096` |
| T08 | `RBS=4; NSP=2; GBS=8; MAX_TOKENS=8192; RESP=3584; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` |

约束：

- `GBS <= RBS * NSP`。
- 8 卡 TP=2 默认 DP=4，`RBS * NSP` 至少为 4。
- T06/T07/T08 是主搜索档；T02/T04/T05 只做回退定位。

## 3. 通用清理命令

每次 OOM、hang、手动中断后执行：

```bash
/workspace/vime-rlk-env/bin/ray stop --force || true
pkill -9 -f "[v]llm serve" || true
pkill -9 -f "[V]LLM::" || true
pkill -9 -f "[r]ay" || true
pkill -9 -f "[p]ython.*train.py" || true
pkill -9 -f "[r]edis" || true
sleep 3
nvidia-smi
```

## 4. Smoke

先跑 T01 candidate full-gradient：

```bash
TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T01 cuda
```

检查：

```bash
rg -n "Using RL-Kernel linear_logp op|fused-tile|rl_kernel_fallback_count|rl_kernel_linear_logp|perf 11|step 11" \
  /workspace/logs/8gpu_T01_full_cuda_*/ray_job_cuda.log
```

通过标准：

- 完成 rollout 0-11。
- 出现 `Using RL-Kernel linear_logp op`。
- 出现 full-gradient fast path。
- `train/rl_kernel_fallback_count` 为 0。

## 5. 最大不 OOM 搜索

搜索顺序：

```text
T01 -> T03 -> T06 -> T07
```

分支规则：

| 结果 | 下一步 |
| --- | --- |
| T03 OOM | 跑 T02；T02 成功后补 T02 baseline |
| T03 成功，T06 OOM | 跑 T05；T05 成功则补 T05 baseline，否则补 T03 baseline |
| T06 成功，T07 成功 | T07 是 candidate 最大候选，补 T07 baseline |
| T06 成功，T07 OOM | 跑 T08；T08 成功则补 T08 baseline，否则补 T06 baseline |

命令：

```bash
TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T03 cuda

TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T06 cuda

TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T07 cuda
```

T06/T07 是最能体现 kernel 性能的配置：

- `RESP` 长，接近模型 `seq_length=4096` 上限。
- `MAX_TOKENS_PER_GPU=8192`，允许动态 batch 把更多 packed tokens 放进一次 `linear_logp`。
- `RBS=4, NSP=2, GBS=8`，每轮 8 条样本，满足 8 卡 TP=2 的 DP=4 调度要求，也避免 `GBS > RBS*NSP`。
- `VLLM_MEM` 没有盲目拉高，避免 rollout KV cache 抢训练显存。

## 6. 训推 trace

只抓 actor train：

```bash
TRACE_MODE=train TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T06 cuda
```

只抓 rollout generate：

```bash
TRACE_MODE=rollout TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T06 cuda
```

训推都抓：

```bash
TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T06 cuda
```

多轮 trace：

```bash
TRACE_MODE=all TRACE_ROLLOUTS=3-5 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T06 cuda
```

说明：

- `TRACE_MODE=train` 捕获 actor train 的 `cudaProfilerStart/Stop` range。
- `TRACE_MODE=rollout` 会自动设置 `VLLM_PROFILER_CONFIG='{"profiler":"cuda"}'`，并在目标 rollout 前后调用 vLLM `/start_profile` 和 `/stop_profile`。
- `TRACE_MODE=all` 两者都捕获。
- Nsight 使用 `--capture-range-end=repeat:64`，能覆盖多个 rollout 的多个 capture range。
- rollout trace 里看 vLLM worker 的 NVTX annotation，例如 `execute_context_*_generation_*`。

## 7. 补 baseline

只给最大 candidate 成功档和一个中等档补 baseline。

示例：最大成功档是 T06。

```bash
TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T06 baseline

TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T06 cuda
```

如果 baseline OOM：

1. 记录当前档为 `baseline OOM / candidate success`。
2. 回退到上一档补 baseline。
3. 在矩阵里同时写 candidate 最大成功档和 baseline 最大成功档。

## 8. 可选 output-layer-only

只在 full-gradient 主结果完成后跑最大成功档。

```bash
TRAIN_SCOPE=output_layer TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T06 baseline

TRAIN_SCOPE=output_layer TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T06 cuda
```

## 9. 结果提取

确认 12 轮完成：

```bash
rg -n "step 11:|perf 11:" /workspace/logs/<RUN>/ray_job_*.log
```

baseline 核心字段：

```bash
rg -n "step [3-9]:|step 1[01]:|perf [3-9]:|perf 1[01]:|baseline_linear_logp|train_rollout_logprob_abs_diff|rollout_time|raw_reward" \
  /workspace/logs/<RUN>/ray_job_baseline.log
```

candidate 核心字段：

```bash
rg -n "step [3-9]:|step 1[01]:|perf [3-9]:|perf 1[01]:|Using RL-Kernel linear_logp op|fused-tile|save-probs|rl_kernel_fallback_count|rl_kernel_linear_logp|train_rollout_logprob_abs_diff|rollout_time|raw_reward" \
  /workspace/logs/<RUN>/ray_job_cuda.log
```

填表规则：

- `run_status`：有 `step 11` 和 `perf 11` 才填 success。
- 时间类指标：取 rollout 3-11 均值。
- 显存峰值：取 rollout 0-11 最大值。
- CUDA event 秒转毫秒：`value * 1000`。
- `tokens/call`：填 `*_tokens_per_call_delta`。
- `token_count_delta`：填 `*_token_count_delta`。
- fallback：candidate 必须是 0。

full-gradient 主指标：

```text
train/baseline_linear_logp_forward_backward_cuda_event_elapsed_s_delta
train/rl_kernel_linear_logp_forward_backward_cuda_event_elapsed_s_delta
```

output-layer-only 主指标：

```text
train/baseline_linear_logp_forward_cuda_event_elapsed_s_delta
train/rl_kernel_linear_logp_forward_cuda_event_elapsed_s_delta
```

## 10. OOM 记录

OOM 时记录：

| 字段 | 内容 |
| --- | --- |
| 配置ID | T01/T03/T05/T06/T07/T08 |
| mode | baseline 或 candidate |
| 触发点 | rollout init / rollout generate / weight sync / train forward / train backward / optimizer |
| 完成轮数 | 最后一个 `step N` 或 `perf N` |
| 最后一条 log | OOM 前最后 20 行 |
| GPU 峰值 | 日志 peak reserved 或 `nvidia-smi` |
| 下一步 | 回退到哪个配置 |

常见 OOM grep：

```bash
rg -n "out of memory|CUDA error|RayOutOfMemoryError|WorkerCrashedError|NCCL|killed|ActorDiedError" /workspace/logs/<RUN>
```

每次 OOM 后执行第 3 节清理命令。
