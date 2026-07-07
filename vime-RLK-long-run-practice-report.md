# vime + RL-Kernel 8 卡长跑实践记录

日期：2026-07-07 UTC

本文记录本轮围绕 `linear_logp` 的 8xH100 长时间验证经验。范围包括环境搭建、分支和运行方式、T01/T03/T06 已完成结果、T07 trace 尝试状态、调试中做过的代码级改动，以及 baseline 与 candidate 算子实现的差异。

本文只纳入 `TRACE_MODE=none` 的完整 12 轮结果做性能比较。trace run 只用于诊断，不混入 timing 表。

## 实验目标

本轮目标不是单独跑一个 microbenchmark，而是在 vime 的完整训推链路里验证 RL-Kernel `linear_logp`：

- 模型：Qwen3-30B-A3B。
- 机器：8x H100 80GB。
- vime 分支：`pr-3`。
- RL-Kernel 分支：PR 211 对应分支。
- 训练并行：TP=2、PP=1、CP=1、EP=8。
- rollout：完整 vLLM rollout，`NUM_ROLLOUT=12`，不使用 debug rollout。
- 对比：
  - baseline：Megatron/vime 原生 output layer + 原生 selected logprob。
  - candidate：RL-Kernel `FusedLinearLogpSM90Op`，`save-logits` / fused-tile full-gradient path。

本文里的 candidate 特指 `save-logits` / full-gradient fused-tile 路径。它对应 `TRAIN_SCOPE=full`、`RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16=0`、`RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL=1`，日志应出现 `Using fused-tile bf16 full-gradient tensor-parallel linear_logp fast path.`。它不是 output-layer-only 的 `save-prob` 路径，也不是一个单独叫 `save-logp` 的模式。

最终要回答两个问题：

1. candidate 在真实 8 卡训推流程中是否稳定完成 12 轮，是否 fallback=0。
2. candidate 的 `linear_logp` 单算子是否比 baseline 更快、显存更低，并解释为什么。

## 工作区和环境

用户要求“环境尽可能装在 workspace”，本轮按这个原则处理：

| 项 | 路径或设置 |
| --- | --- |
| workspace | `/workspace` |
| vime | `/workspace/vime` |
| RL-Kernel | `/workspace/RL-Kernel` |
| Megatron-LM | `/workspace/Megatron-LM` |
| Python env | `/workspace/vime-rlk-env` |
| HF cache | `/workspace/.cache/huggingface` |
| pip cache | `/workspace/.cache/pip` |
| torch cache | `/workspace/.cache/torch` |
| temp dir | `/workspace/.cache/tmp` |
| FlashInfer workspace | `/workspace/.cache/flashinfer` |
| logs | `/workspace/logs` |
| traces | `/workspace/nsys_traces` |

脚本里已经把这些 cache 环境变量写入 workspace：

```bash
export HF_HOME="${WORKSPACE_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR="${WORKSPACE_ROOT}/.cache/pip"
export TORCH_HOME="${WORKSPACE_ROOT}/.cache/torch"
export XDG_CACHE_HOME="${WORKSPACE_ROOT}/.cache"
export TMPDIR="${WORKSPACE_ROOT}/.cache/tmp"
export FLASHINFER_WORKSPACE_BASE="${WORKSPACE_ROOT}"
export FLASHINFER_CUDA_ARCH_LIST="9.0"
```

这点很重要：Qwen3-30B-A3B、vLLM、FlashInfer、torch extension 会产生较多缓存和临时文件，如果落到默认 HOME 或系统临时目录，长跑中更容易遇到空间、权限或不可复现问题。

## 运行入口

主入口是：

```bash
cd /workspace/vime
WORKSPACE_ROOT=/workspace \
VIME_PYTHON_ENV=/workspace/vime-rlk-env \
TRACE_MODE=none \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T01 cuda
```

其中：

- `MODE=cuda` 表示 candidate，开启 RL-Kernel `linear_logp`。
- `MODE=baseline` 表示关闭 RL-Kernel，打开 baseline timer。
- `TRACE_MODE=none` 是正式性能数据口径。
- `TRACE_MODE=all TRACE_ROLLOUTS=3` 只用于单独抓 trace。

candidate full-gradient 的关键环境变量：

```bash
export VIME_RL_KERNEL=1
export VIME_RL_KERNEL_OPS=linear_logp
export VIME_RL_KERNEL_LINEAR_LOGP_BACKEND=cuda
export VIME_RL_KERNEL_CUDA_EVENT_TIMER=1
export VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN=0
export RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16=0
export RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL=1
```

baseline 的关键环境变量：

```bash
export VIME_RL_KERNEL=0
export VIME_BASELINE_LINEAR_LOGP_TIMER=1
export VIME_BASELINE_CUDA_EVENT_TIMER=1
export RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16=0
export RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL=0
```

清理残留进程的实际经验：

```bash
/workspace/vime-rlk-env/bin/ray stop --force
pkill -f 'vllm|ray::|redis-server|train'
nvidia-smi
```

T03 baseline 第一次失败后，必须清理 Ray/vLLM/redis 残留再重跑，否则下一次 run 容易继承坏状态。

## 完整操作步骤

这一节按“从空闲机器到拿到一组可用 metrics”的顺序写。命令默认在 `/workspace` 下执行，除非特别说明。

### 1. 确认分支和工作树

先确认三个仓库的分支状态。不要在未确认分支时直接开跑，因为 vime PR、RL-Kernel PR 和 Megatron 本地兼容补丁缺一项都可能导致数据不可复现。

```bash
git -C /workspace/vime status --short --branch
git -C /workspace/RL-Kernel status --short --branch
git -C /workspace/Megatron-LM status --short --branch
```

本轮期望状态：

```text
/workspace/vime       branch: pr-3
/workspace/RL-Kernel  branch: pr-211 / origin/pr/211
/workspace/Megatron-LM local compatibility patches present
```

如果分支不对，按下面方式切换：

```bash
cd /workspace/vime
gh pr checkout 3

cd /workspace/RL-Kernel
gh pr checkout 211
```

注意事项：

- 不要把 GitHub token 写入文档、日志或 shell history。
- `/workspace/Megatron-LM` 当前是 detached/local patch 状态，不要用 `git reset --hard` 清掉；这些补丁用于本环境跑通 Qwen3-MoE。
- 如果 `git status` 里已有用户改动，提交 PR 前单独确认，不要为了跑实验而回滚。

### 2. 激活 workspace 内环境

本轮使用 `/workspace/vime-rlk-env`，并让 cache、临时目录、FlashInfer 编译产物尽量落到 `/workspace`。

```bash
export WORKSPACE_ROOT=/workspace
export VIME_PYTHON_ENV=/workspace/vime-rlk-env
export PATH="${VIME_PYTHON_ENV}/bin:${PATH}"

export HF_HOME="${WORKSPACE_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export PIP_CACHE_DIR="${WORKSPACE_ROOT}/.cache/pip"
export TORCH_HOME="${WORKSPACE_ROOT}/.cache/torch"
export XDG_CACHE_HOME="${WORKSPACE_ROOT}/.cache"
export TMPDIR="${WORKSPACE_ROOT}/.cache/tmp"
export FLASHINFER_WORKSPACE_BASE="${WORKSPACE_ROOT}"
export FLASHINFER_CUDA_ARCH_LIST=9.0

mkdir -p \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${PIP_CACHE_DIR}" \
  "${TORCH_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${TMPDIR}" \
  "${FLASHINFER_WORKSPACE_BASE}/.cache/flashinfer"
```

验证 Python 和包路径：

```bash
which python
python - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY
```

期望 `sys.executable` 是：

```text
/workspace/vime-rlk-env/bin/python
```

### 3. 确认 RL-Kernel CUDA extension 可用

candidate 必须命中 `FusedLinearLogpSM90Op`，所以开跑前先确认 extension 里有 SM90 forward/backward 符号。

```bash
cd /workspace/RL-Kernel
python - <<'PY'
from rl_engine.kernels.ops.cuda.loss.linear_logp import _C
print("fused_fwd", hasattr(_C, "fused_linear_logp_sm90"))
print("fused_bwd", hasattr(_C, "fused_linear_logp_sm90_backward"))
print("tp_global_fwd", hasattr(_C, "fused_linear_logp_sm90_global_target"))
print("save_probs_fwd", hasattr(_C, "linear_logp_probs_bf16_forward"))
PY
```

至少需要：

```text
fused_fwd True
fused_bwd True
```

如果这里失败，先不要跑 8 卡。通常处理顺序是：

```bash
cd /workspace/RL-Kernel
KERNEL_ALIGN_FORCE_SM90=1 pip install -e .
```

然后重新执行上面的 Python 检查。

### 4. 清理上一轮残留进程

每次完整 8 卡 run 之前都要清理 Ray、vLLM、redis 和训练进程。尤其是 EngineDead、OOM、中断 trace 之后，残留进程会导致下一轮端口冲突、GPU 显存残留或 Ray object store 状态污染。

```bash
/workspace/vime-rlk-env/bin/ray stop --force || true
pkill -f 'vllm|ray::|redis-server|train' || true
sleep 5
nvidia-smi
```

理想状态是每张 H100 只剩很低的 context 占用，例如几 MB。如果某张卡还有几十 GB，说明仍有进程没杀掉，需要用：

```bash
nvidia-smi
ps -ef | rg 'vllm|ray|python|train'
```

定位 PID 后再处理。

### 5. 确认 benchmark 脚本的配置展开

入口脚本是：

```text
/workspace/vime/scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh
```

脚本先根据 `CONFIG_ID` 展开 T01/T03/T06/T07 的关键配置。T07 当前配置在脚本里等价于：

```bash
ROLLOUT_BATCH_SIZE=4
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=8
MAX_TOKENS_PER_GPU=8192
ROLLOUT_MAX_RESPONSE_LEN=3968
VLLM_GPU_MEMORY_UTILIZATION=0.46
VLLM_MAX_MODEL_LEN=4096
```

并统一设置：

```bash
NUM_GPUS=8
MEGATRON_TP=2
MEGATRON_EP=8
MEGATRON_CP=1
ROLLOUT_NUM_GPUS_PER_ENGINE=8
NUM_ROLLOUT=12
VIME_DISABLE_SAVE=1
VIME_SKIP_EVAL_BEFORE_TRAIN=1
VIME_USE_KL_LOSS=0
VIME_SKIP_ZERO_ENTROPY_METRIC=1
VIME_VLLM_ENFORCE_EAGER=1
VLLM_WORKER_MULTIPROC_METHOD=spawn
VIME_TRAIN_MEMORY_MARGIN_BYTES=1073741824
VIME_SKIP_MOE_GROUPED_GEMM_CAPABILITY_CHECK=1
MEGATRON_ALLOW_MOE_TP_WITHOUT_SP=1
CUDA_MODULE_LOADING=EAGER
```

这些环境变量不是装饰项：

- `VIME_SKIP_ZERO_ENTROPY_METRIC=1` 让 `entropy_coef=0` 时不计算无用 entropy，避免 candidate 因 entropy fallback。
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` 避免 vLLM worker 继承复杂父进程 CUDA 状态。
- `VIME_SKIP_MOE_GROUPED_GEMM_CAPABILITY_CHECK=1` 只绕过 Megatron validation 阶段的 capability check，runtime 仍保留 grouped GEMM。
- `MEGATRON_ALLOW_MOE_TP_WITHOUT_SP=1` 是本地 Megatron 兼容补丁的开关。

### 6. candidate 与 baseline 在脚本里的差异

脚本根据第二个参数 `MODE` 设置两套路径。

candidate：

```bash
MODE=cuda
VIME_RL_KERNEL=1
VIME_BASELINE_LINEAR_LOGP_TIMER=0
VIME_BASELINE_CUDA_EVENT_TIMER=0
VIME_RL_KERNEL_LINEAR_LOGP_BACKEND=cuda
VIME_RL_KERNEL_CUDA_EVENT_TIMER=1
VIME_RL_KERNEL_LINEAR_LOGP_DETACH_HIDDEN=0
RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16=0
RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL=1
```

baseline：

```bash
MODE=baseline
VIME_RL_KERNEL=0
VIME_BASELINE_LINEAR_LOGP_TIMER=1
VIME_BASELINE_CUDA_EVENT_TIMER=1
RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16=0
RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL=0
```

full-gradient 主结果都使用：

```bash
TRAIN_SCOPE=full
```

`output_layer` scope 只用于可选补充，不作为本轮主结论。

### 7. 启动正式 no-trace run

正式性能 run 必须用：

```bash
TRACE_MODE=none
```

T07 candidate 本轮重跑命令如下：

```bash
cd /workspace/vime
WORKSPACE_ROOT=/workspace \
RUN_ROOT=/workspace \
VIME_PYTHON_ENV=/workspace/vime-rlk-env \
TRACE_MODE=none \
TRAIN_SCOPE=full \
RUN_NAME=8gpu_T07_full_cuda_20260707_155623 \
VIME_UPDATE_WEIGHT_BUFFER_SIZE=134217728 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T07 cuda
```

预期日志路径：

```text
/workspace/logs/8gpu_T07_full_cuda_20260707_155623/ray_job_cuda.log
```

为什么设置 `VIME_UPDATE_WEIGHT_BUFFER_SIZE=134217728`：

- 这个值是 128MiB。
- 8 卡长跑中 update weight 是固定开销之一。
- 分桶太大容易让显存峰值和 Ray object 传输抖动变大。
- 分桶日志可以帮助区分“算子慢”和“权重同步慢”。

### 8. 启动后第一轮必须检查的日志

启动后先确认 Ray job 进入训练，而不是卡在环境安装或 vLLM 启动。

```bash
tail -f /workspace/logs/8gpu_T07_full_cuda_20260707_155623/ray_job_cuda.log
```

另开一个 shell 用 `rg` 查关键行：

```bash
LOG=/workspace/logs/8gpu_T07_full_cuda_20260707_155623/ray_job_cuda.log
rg -n \
  "Using RL-Kernel linear_logp op|fused-tile|rl_kernel_fallback_count|perf [0-9]+:|step [0-9]+:|train_rollout_logprob_abs_diff|CUDA out of memory|EngineDead|ConnectionRefusedError" \
  "$LOG"
```

必须看到：

```text
Using RL-Kernel linear_logp op: FusedLinearLogpSM90Op
Using fused-tile bf16 full-gradient tensor-parallel linear_logp fast path.
```

并且每轮 `fallback` 必须保持 0。若出现 fallback 非 0，这组 candidate 数据不能作为宣传结果，需要先 debug。

### 9. 代码级路径：candidate 怎么绕开完整 logits

T07 candidate run 的关键代码路径如下。

第一步，vime 根据参数启用 RL-Kernel：

```text
/workspace/vime/scripts/run-qwen3-30B-A3B.sh
```

脚本把参数传进 Ray runtime env：

```bash
RLK_ARGS+=(--enable-rl-kernel --rl-kernel-ops "${VIME_RL_KERNEL_OPS:-linear_logp}")
```

第二步，Megatron model 构建后，vime 从 model 里取 output layer 权重和 TP 信息：

```text
/workspace/vime/vime/backends/megatron_utils/rl_kernel.py
get_linear_logp_context_from_model()
```

核心字段：

```python
LinearLogpContext(
    lm_head_weight=weight,
    bias=bias,
    tp_group=tp_group,
    vocab_start_index=vocab_start_index,
    global_vocab_size=global_vocab_size,
    sequence_parallel=...,
)
```

第三步，前向时临时让 Megatron 返回 hidden states，而不是让 output layer 直接产出 logits：

```python
@contextmanager
def return_hidden_states_for_linear_logp(args, model, context):
    old_post_process = module.post_process
    module.post_process = False
    try:
        yield True
    finally:
        module.post_process = old_post_process
```

第四步，loss 侧把 hidden states 展平，构建 shifted target token，然后调用：

```text
/workspace/vime/vime/backends/megatron_utils/loss.py
get_log_probs_and_entropy()
```

关键分支：

```python
if linear_logp_context is not None:
    log_prob_full = maybe_compute_linear_logp(
        logits,
        full_tokens,
        context=linear_logp_context,
        args=args,
        with_entropy=with_entropy,
    )
```

这里变量名仍叫 `logits`，但在 candidate 路径它实际是 hidden states；如果 RL-Kernel 不能使用，才会调用 `_materialize_linear_logits()` 退回完整 logits。

第五步，真正调用 RL-Kernel：

```text
/workspace/vime/vime/backends/megatron_utils/rl_kernel.py
maybe_compute_linear_logp()
```

核心调用：

```python
log_prob = op(
    hidden_states,
    weight,
    target_ids.long(),
    bias,
    tp_group=context.tp_group,
    vocab_start_index=context.vocab_start_index,
    global_vocab_size=context.global_vocab_size,
)
```

第六步，RL-Kernel 选择 SM90 fused path：

```text
/workspace/RL-Kernel/rl_engine/kernels/ops/cuda/loss/linear_logp.py
FusedLinearLogpSM90Op.apply()
```

T07 full-gradient 期望命中：

```python
return _TensorParallelLinearLogpFusedTileBF16Function.apply(
    hidden,
    lm_head_weight,
    target_ids,
    int(vocab_start_index),
    None if global_vocab_size is None else int(global_vocab_size),
    tp_group,
)
```

第七步，CUDA extension forward：

```text
/workspace/RL-Kernel/csrc/cuda/fused_linear_logp_sm90.cu
fused_linear_logp_sm90_forward_impl()
```

这个函数做的事：

- 检查 hidden/weight 是 CUDA bf16 contiguous tensor。
- 用 TMA descriptor 读取 hidden tile 和 weight tile。
- 在 tile GEMM 中累计 local max、sum exp、target logit。
- 返回 `[N]` 级别的 `out_value` 和 `lse`，不返回 `[N, V]` logits。

第八步，CUDA extension backward：

```text
/workspace/RL-Kernel/csrc/cuda/fused_linear_logp_sm90.cu
fused_linear_logp_sm90_backward()
```

T07 full-gradient 需要 `compute_grad_hidden` 和 `compute_grad_weight`，因此重点看 full fused tile branch：

```cpp
if ((full_fused_tile_mode == "tile_cublas" || full_fused_tile_mode == "tile" ||
     full_fused_tile_mode == "streaming" || full_fused_tile_mode == "tiled") &&
    (compute_grad_hidden || compute_grad_weight) && !compute_grad_bias &&
    !bias.has_value() && hidden.scalar_type() == at::kBFloat16 &&
    weight.scalar_type() == at::kBFloat16 && D % BK == 0) {
    ...
}
```

### 10. 代码级路径：baseline 为什么更重

baseline 不启用 `linear_logp_context`，所以 `get_log_probs_and_entropy()` 进入原生路径：

```python
if linear_logp_context is None:
    logits = logits.contiguous()
    log_prob_full, entropy_full = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
    )
```

对应：

```text
/workspace/vime/vime/utils/ppo_utils.py
calculate_log_probs_and_entropy()
```

核心逻辑：

```python
log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
```

这就是 baseline 的主要额外成本：

- output layer 已经物化 `[T, V]` logits。
- logprob 又 clone 一份 logits。
- softmax/logsumexp/gather 在完整 vocab 维度上做。
- backward 也围绕完整 logits 图回传。

baseline 计时 hook 在：

```text
/workspace/vime/vime/backends/megatron_utils/model.py
_probe_baseline_output_layer_forward()
```

它记录 output layer 的 forward 和 forward+backward CUDA event，再与 native logprob timer 相加，形成：

```text
train/baseline_linear_logp_forward_cuda_event_elapsed_s_delta
train/baseline_linear_logp_forward_backward_cuda_event_elapsed_s_delta
train/baseline_linear_logp_dispatch_elapsed_s_delta
```

### 11. 每轮判定标准

一轮可接受 candidate 结果至少满足：

```text
run_status: success
rl_kernel_fallback_count_delta: 0
rl_kernel_linear_logp_call_count_delta > 0
rl_kernel_linear_logp_token_count_delta > 0
train_rollout_logprob_abs_diff: finite, same order as previous runs
loss/reward: finite
```

T01/T03/T06 已完成结果的 abs diff 参考：

```text
T01 candidate abs diff mean 3-11: 0.02537
T03 candidate abs diff mean 3-11: 0.02395
T06 candidate abs diff mean 3-11: 0.02224
```

T07 如果 abs diff 是同一量级，可以认为 correctness 没有明显异常；如果突然变成很大或 NaN，要先查 rollout/train logprob 对齐和 target 构造。

### 12. metrics 提取步骤

正式统计使用 rollout 3-11 均值，避开前 3 轮 warmup。

T07 完成后先查这些行：

```bash
LOG=/workspace/logs/8gpu_T07_full_cuda_20260707_155623/ray_job_cuda.log

rg -n \
  "perf [3-9]:|perf 1[01]:|step [3-9]:|step 1[01]:|rl_kernel_linear_logp|train_rollout_logprob_abs_diff|raw_reward|peak_reserved|fallback" \
  "$LOG"
```

需要写入最终矩阵的字段：

```text
run_status
peak_vram_gb
step_time_s mean 3-11
train_time_s mean 3-11
actor_train_time_s mean 3-11
rollout_time_s mean 3-11
tokens_per_gpu_per_sec mean 3-11
raw_reward mean 3-11
train_rollout_logprob_abs_diff mean 3-11
rl_kernel_linear_logp_tokens_per_call_delta mean 3-11
rl_kernel_linear_logp_forward_cuda_event_elapsed_s_delta mean 3-11
rl_kernel_linear_logp_forward_backward_cuda_event_elapsed_s_delta mean 3-11
rl_kernel_linear_logp_dispatch_elapsed_s_delta mean 3-11
peak_alloc_delta MB
peak_reserved_delta MB
fallback
```

如果 T07 candidate 成功，再决定是否补 baseline。baseline 命令为：

```bash
cd /workspace/vime
WORKSPACE_ROOT=/workspace \
RUN_ROOT=/workspace \
VIME_PYTHON_ENV=/workspace/vime-rlk-env \
TRACE_MODE=none \
TRAIN_SCOPE=full \
RUN_NAME=8gpu_T07_full_baseline_YYYYMMDD_HHMMSS \
VIME_UPDATE_WEIGHT_BUFFER_SIZE=134217728 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T07 baseline
```

baseline 完成后再用同样 rollout 3-11 口径更新 `vime-RLK-final-metrics-config-matrix.md`。

### 13. T07 重新启动前的当前阻塞状态

2026-07-07 15:59 UTC 准备按以下命令重新启动 T07 candidate no-trace：

```bash
cd /workspace/vime
WORKSPACE_ROOT=/workspace \
RUN_ROOT=/workspace \
VIME_PYTHON_ENV=/workspace/vime-rlk-env \
TRACE_MODE=none \
TRAIN_SCOPE=full \
RUN_NAME=8gpu_T07_full_cuda_20260707_155623 \
VIME_UPDATE_WEIGHT_BUFFER_SIZE=134217728 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T07 cuda
```

启动前检查发现每卡仍有约 41GB 显存占用：

```text
GPU0 41569 MiB
GPU1 41713 MiB
GPU2 41713 MiB
GPU3 41713 MiB
GPU4 41713 MiB
GPU5 41713 MiB
GPU6 41713 MiB
GPU7 40993 MiB
```

`nvidia-smi` 主进程表和 `nvidia-smi pmon` 没列出进程，但 NVML compute-app query 能看到宿主侧不可见 PID：

```text
2195449 [Not Found] GPU0 41560 MiB
2195450 [Not Found] GPU1 41704 MiB
2195451 [Not Found] GPU2 41704 MiB
2195452 [Not Found] GPU3 41704 MiB
2195453 [Not Found] GPU4 41704 MiB
2195454 [Not Found] GPU5 41704 MiB
2195455 [Not Found] GPU6 41704 MiB
2195456 [Not Found] GPU7 40984 MiB
```

在当前容器里 `/proc/2195449` 到 `/proc/2195456` 不存在，`kill -9` 返回 `No such process`，说明这些 PID 不在当前 PID namespace。`nvidia-smi --gpu-reset -i 0,1,2,3,4,5,6,7` 返回 `Not Supported`。因此 T07 暂时不能安全启动；否则会在已有 41GB 占用上叠加训练/rollout 显存，基本确定 OOM。

处理建议：

1. 在宿主机或拥有宿主 PID namespace 的管理端 kill `2195449-2195456`。
2. 或释放对应占用 GPU 的外部容器/作业。
3. 释放后确认 `nvidia-smi --query-gpu=index,memory.used` 每卡回到几 MB。
4. 再执行上面的 T07 candidate no-trace 命令。

### 14. 异常处理步骤

如果 T07 candidate OOM：

1. 保存 log，不覆盖。
2. 记录最后一轮 peak reserved 和 OOM 栈。
3. 清理 Ray/vLLM 进程。
4. 回退 T08：`RESP=3584`，`VLLM_MEM=0.45`。
5. 不补 T07 baseline，除非需要证明 baseline 也 OOM。

如果 T07 candidate fallback 非 0：

1. 查是否缺少 `Using fused-tile bf16 full-gradient tensor-parallel linear_logp fast path.`。
2. 查 hidden/weight dtype 是否仍是 bf16。
3. 查 bias 是否变成非 None。
4. 查 `with_entropy` 是否被打开。
5. 查 `VIME_SKIP_ZERO_ENTROPY_METRIC=1` 是否传进 Ray runtime env。
6. 查 `RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL=1` 是否生效。

如果 vLLM wakeup 或 onload weights 失败：

1. 先看是否有 `ConnectionRefusedError`、`EngineDead`、`APIServer` 退出。
2. 不要直接重跑同一个 shell；先执行清理命令。
3. 检查端口和 GPU 进程。
4. 重跑同一配置，失败 run 单独标记为 startup failure，不纳入 metrics。

如果 candidate 完整 step 慢于 baseline：

1. 先比较单算子 fwd/fwd+bwd，而不是 step time。
2. 查 rollout time、weight-sync bucket time、Ray object store 和 vLLM wakeup。
3. 查 tokens/call 是否太小。
4. 查是否开了 trace。
5. 查 baseline 是否包含 spike，不能用异常 spike 夸大或误判。

## RL 框架级工作流程

这一节按 vime 本轮 8 卡 colocate 训练的实际代码路径讲清楚 RL 框架如何工作。重点不是命令行参数，而是 Ray、vLLM、Megatron、loss、权重同步之间的数据和控制流。

本轮入口脚本最终提交的 Ray job 是：

```bash
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${NUM_GPUS} \
  --colocate \
  ...
```

所以本轮主流程在：

```text
/workspace/vime/train.py
```

不是 `train_async.py`。`train_async.py` 明确 `assert not args.colocate`，而本轮使用 `--colocate`，训练和 rollout 共享同一组 8 张 H100，通过 sleep/wake 和 weight sync 协调显存。

### 总体循环

`train.py` 的主循环可以抽象成：

```python
def train(args):
    pgs = create_placement_groups(args)
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        rollout_manager.onload_weights()

    actor_model.update_weights()

    if args.offload_rollout:
        rollout_manager.onload_kv()

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        rollout_data_ref = rollout_manager.generate.remote(rollout_id)

        if args.offload_rollout:
            rollout_manager.offload()

        actor_model.async_train(rollout_id, rollout_data_ref)

        if args.offload_rollout:
            rollout_manager.onload_weights()

        actor_model.update_weights()

        if args.offload_rollout:
            rollout_manager.onload_kv()
```

真实代码里还有 eval、save、critic、global dataset、periodic action，但 T01/T03/T06/T07 这条主线就是：

```text
启动 Ray/vLLM/Megatron
-> 初始 Megatron actor weights 推给 vLLM
-> vLLM rollout 生成 samples
-> samples 转成 Megatron train batch
-> Megatron actor train
-> Megatron 新权重同步到 vLLM
-> 下一轮 rollout
```

### Ray placement group：谁占哪张 GPU

代码位置：

```text
/workspace/vime/vime/ray/placement_group.py
create_placement_groups()
create_rollout_manager()
create_training_models()
```

本轮 `--colocate` 为 True，因此：

```python
elif args.colocate:
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    rollout_offset = 0
```

含义：

- Ray 只创建一个包含 8 个 GPU bundle 的 placement group。
- actor 和 rollout 都从同一个 placement group 里取资源。
- rollout offset 是 0，说明 vLLM engine 和 Megatron actor 逻辑上共享同一批 GPU。

`_create_placement_group()` 做两件关键事：

1. 创建 `bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]`。
2. 用临时 `InfoActor` 查询每个 bundle 实际落在哪个 node/GPU，然后按 node 和 GPU ID 排序，得到稳定的 `pg_reordered_bundle_indices` 和 `pg_reordered_gpu_ids`。

这一步重要是因为 Ray 的 bundle index 不一定天然等于物理 GPU ID。后续 vLLM 的 `base_gpu_id`、Megatron rank 和 colocate weight sync 都依赖这个排序。

### RolloutManager：vLLM rollout 的控制面

代码位置：

```text
/workspace/vime/vime/ray/rollout.py
RolloutManager
RolloutServer
ServerGroup
```

`create_rollout_manager()` 创建一个 Ray actor：

```python
rollout_manager = RolloutManager.options(
    num_cpus=1,
    num_gpus=0,
).remote(args, pg)
```

`RolloutManager.__init__()` 做的事：

1. 加载数据源：

   ```python
   data_source_cls = load_function(self.args.data_source_path)
   self.data_source = data_source_cls(args)
   ```

2. 加载 rollout 函数：

   ```python
   self.generate_rollout = load_function(self.args.rollout_function_path)
   self.eval_generate_rollout = load_function(self.args.eval_function_path)
   ```

3. 如果不是 debug train-only，就启动 vLLM servers：

   ```python
   self.servers = start_rollout_servers(args, pg)
   ```

4. 创建 rollout engine lock：

   ```python
   self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
   ```

这个 lock 在权重同步时用来协调 rollout engines，不让 vLLM 在更新权重时同时生成。

### ServerGroup：如何启动 vLLM engine

代码位置：

```text
/workspace/vime/vime/ray/rollout.py
ServerGroup.start_engines()
```

每个 `ServerGroup` 表示一组同构 vLLM engine。T01/T03/T06/T07 这种单模型单 engine 配置通常只有一个主要 server group。

启动逻辑：

```python
RolloutRayActor = ray.remote(VLLMEngine)

rollout_engine = RolloutRayActor.options(
    num_cpus=num_cpus,
    num_gpus=0.2,
    scheduling_strategy=PlacementGroupSchedulingStrategy(...),
    runtime_env={"env_vars": env_vars},
).remote(
    self.args,
    rank=global_rank,
    worker_type=self.worker_type,
    base_gpu_id=base_gpu_id,
    vllm_overrides=self.vllm_overrides,
    num_gpus_per_engine=self.num_gpus_per_engine,
)
```

这里 Ray actor 只申请 `num_gpus=0.2`，不是因为 vLLM 只用 0.2 张卡，而是因为真正的 vLLM server 是 actor 里再 spawn 出来的子进程。Ray 资源只是占位和调度，实际 CUDA 可见设备通过 `CUDA_VISIBLE_DEVICES` 控制。

`base_gpu_id` 来自 placement group 的排序结果，表示该 vLLM engine 从哪张物理 GPU 开始取连续设备。

### VLLMEngine：Ray actor 到 vLLM server 子进程

代码位置：

```text
/workspace/vime/vime/backends/vllm_utils/vllm_engine.py
VLLMEngine.init()
launch_server_process()
_build_subprocess_env()
_run_vllm_server()
```

`VLLMEngine.init()` 先计算 server args：

```python
server_args_dict, external_engine_need_check_fields = _compute_server_args(...)
```

然后普通本地模式走：

```python
self._init_normal(server_args_dict)
```

`_init_normal()` 里真正启动 vLLM：

```python
self.process = launch_server_process(server_args_dict)
```

`launch_server_process()` 做三件关键事：

1. 构造子进程环境：

   ```python
   env = _build_subprocess_env(server_args_dict)
   ```

2. 强制 multiprocessing spawn：

   ```python
   multiprocessing.set_start_method("spawn", force=True)
   p = multiprocessing.Process(target=_run_vllm_server, args=(kwargs, env))
   p.start()
   ```

3. node rank 0 等 `/health`：

   ```python
   _wait_server_healthy(base_url=..., is_process_alive=lambda: p.is_alive())
   ```

`_build_subprocess_env()` 是本轮 debug 的关键点之一：

```python
env["CUDA_VISIBLE_DEVICES"] = server_args_dict["_visible_devices"]
env.setdefault("VLLM_SERVER_DEV_MODE", "1")
env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
env.setdefault("NCCL_CUMEM_ENABLE", "0")
```

含义：

- Ray actor 自己不直接决定 vLLM 用哪些 GPU，vLLM server 子进程通过 `CUDA_VISIBLE_DEVICES` 绑定。
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` 避免 fork 继承父进程复杂 CUDA/Ray 状态。
- colocate 模式会把 vime root 补进 `PYTHONPATH`，并允许 vLLM IPC weight update 的序列化：

  ```python
  env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
  ```

`_run_vllm_server()` 直接调用 vLLM OpenAI server 入口：

```python
from vllm.entrypoints.cli.serve import ServeSubcommand
...
ServeSubcommand.cmd(args)
```

因此 vime 不是手写推理内核，而是把 vLLM 当作一个 HTTP server 管理。

### vLLM sleep/wake：colocate 显存协调

代码位置：

```text
/workspace/vime/vime/backends/vllm_utils/vllm_engine.py
release_memory_occupation()
resume_memory_occupation()
```

vLLM offload：

```python
def release_memory_occupation(self, level: int = 2):
    self.flush_cache()
    response = requests.post(f"http://{self.server_host}:{self.server_port}/sleep", params={"level": level})
```

vLLM onload：

```python
def resume_memory_occupation(self, tags: list[str] = None):
    tags = _normalize_vllm_wake_tags(tags)
    response = requests.post(f"http://{self.server_host}:{self.server_port}/wake_up", params=wake_params)
```

本轮主要用两个 tag：

```text
weights
kv_cache
```

`train.py` 里可以看到顺序：

```python
if args.offload_rollout:
    rollout_manager.onload_weights()

actor_model.update_weights()

if args.offload_rollout:
    rollout_manager.onload_kv()
```

这表示：

1. 先 wake vLLM weights，让它能接收新权重。
2. Megatron actor 把训练后的权重同步到 vLLM。
3. 再 wake KV cache / CUDA graph 等生成所需状态。

训练时则反向：

```python
rollout_data_ref = rollout_manager.generate.remote(rollout_id)

if args.offload_rollout:
    rollout_manager.offload()

actor_model.async_train(...)
```

也就是 rollout 生成结束后先让 vLLM sleep，释放显存给 Megatron train。

### rollout 数据如何变成训练 batch

代码位置：

```text
/workspace/vime/vime/ray/rollout.py
RolloutManager.generate()
RolloutManager._get_rollout_data()
RolloutManager._convert_samples_to_train_data()
RolloutManager._split_train_data_by_dp()
```

`RolloutManager.generate()` 的主逻辑：

```python
data, metrics = self._get_rollout_data(rollout_id=rollout_id)
data = self._convert_samples_to_train_data(data)
return self._split_train_data_by_dp(data)
```

`_get_rollout_data()` 有两种来源：

1. `load_debug_rollout_data`：从磁盘读已保存 rollout。
2. 正常路径：

   ```python
   data = call_rollout_fn(self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False)
   metrics = data.metrics
   data = data.samples
   ```

`generate_rollout` 是用户配置的 rollout function，本轮 Qwen3/vLLM 路径会通过 vLLM HTTP server 生成 response，并形成 `Sample`。

`_convert_samples_to_train_data()` 把 `Sample` 列表转为训练所需字段：

```python
train_data = {
    "tokens": [sample.tokens for sample in samples],
    "response_lengths": [sample.response_length for sample in samples],
    "rewards": rewards,
    "raw_reward": raw_rewards,
    "truncated": ...,
    "sample_indices": ...,
    "rollout_ids": ...,
    "loss_masks": ...,
}
```

几个字段的 RL 含义：

- `tokens`：prompt + response 的完整 token 序列。
- `response_lengths`：只在 response token 上计算 policy loss。
- `loss_masks`：哪些 response token 参与 loss。
- `rewards/raw_reward`：规则 RM 或外部 RM 产生的奖励。
- `rollout_log_probs`：如果 rollout 侧带回生成时 logprob，可用于 off-policy correction 或 mismatch metric。
- `rollout_mask_sums`：同一个 rollout 拆成多个 training samples 时，用于保持“每个 rollout 算一次”的归一化口径。

奖励后处理在：

```python
raw_rewards, rewards = self._post_process_rewards(samples)
```

例如 GRPO/GSPO 下可以做 group normalization：

```python
rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
rewards = rewards - mean
rewards = rewards / (std + 1e-6)
```

最后 `_split_train_data_by_dp()` 根据 DP size 和动态 batch schedule 切分给每个 DP rank：

```python
partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(...)
...
rollout_data_refs.append(Box(ray.put(rollout_data)))
```

这里返回的是一组 Ray object refs，每个 DP rank 一个 `Box(ray.put(...))`。训练 actor 后面会按自己的 DP rank 取对应切片。

### Megatron train actors 如何创建

代码位置：

```text
/workspace/vime/vime/ray/actor_group.py
RayTrainGroup

/workspace/vime/vime/backends/megatron_utils/actor.py
MegatronTrainRayActor
```

`create_training_models()` 创建 actor train group：

```python
actor_model = allocate_train_group(
    args=actor_args,
    num_nodes=args.actor_num_nodes,
    num_gpus_per_node=args.actor_num_gpus_per_node,
    pg=pgs["actor"],
)
```

`RayTrainGroup._allocate_gpus_for_actor()` 中：

```python
TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(MegatronTrainRayActor)
...
actor = TrainRayActor.options(
    num_cpus=num_gpus_per_actor,
    num_gpus=num_gpus_per_actor,
    scheduling_strategy=PlacementGroupSchedulingStrategy(...),
).remote(world_size, rank, master_addr, master_port)
```

每个 GPU 一个 Megatron train actor。Ray 层给 actor 放到对应 bundle 上，Megatron 内部再用 `rank/world_size/master_addr/master_port` 初始化 torch distributed。

传给 train actor 的关键环境变量来自 `scripts/run-qwen3-30B-A3B.sh` 的 Ray runtime env，例如：

```text
PYTHONPATH
HF_HOME / TRANSFORMERS_CACHE / TMPDIR
VIME_RL_KERNEL_LINEAR_LOGP_BACKEND
VIME_RL_KERNEL_CUDA_EVENT_TIMER
VIME_SKIP_ZERO_ENTROPY_METRIC
MEGATRON_LOCAL_ATTENTION_SINGLE_PACKED_SEQ
MEGATRON_ALLOW_MOE_TP_WITHOUT_SP
```

`MegatronTrainRayActor.init()` 做的事：

1. 初始化 torch distributed / Megatron：

   ```python
   monkey_patch_torch_dist()
   super().init(args, role, ...)
   init(args)
   ```

2. 每个 local GPU 依次读 HF config/tokenizer，避免并发写 cache：

   ```python
   for i in range(args.num_gpus_per_node):
       if i == dist.get_rank() % args.num_gpus_per_node:
           self.hf_config = AutoConfig.from_pretrained(...)
           self.tokenizer = AutoTokenizer.from_pretrained(...)
       dist.barrier(group=get_gloo_group())
   ```

3. 构建 Megatron model/optimizer/scheduler：

   ```python
   self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(...)
   ```

4. 记录训练并行配置，给 rollout DP split 使用：

   ```python
   self.train_parallel_config = {
       "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
       "cp_size": mpu.get_context_parallel_world_size(),
       "vpp_size": vpp_size,
       "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
   }
   ```

5. 创建权重备份器：

   ```python
   self.weights_backuper = TensorBackuper.create(...)
   self.weights_backuper.backup("actor")
   ```

6. 根据 colocate 选择权重同步实现：

   ```python
   if self.args.colocate:
       update_weight_cls = UpdateWeightFromTensor
   else:
       update_weight_cls = UpdateWeightFromDistributed
   self.weight_updater = update_weight_cls(...)
   ```

本轮是 colocate，所以走 `UpdateWeightFromTensor`。

### Megatron actor train：从 Ray object 到 GPU tensor

代码位置：

```text
/workspace/vime/vime/backends/megatron_utils/actor.py
MegatronTrainRayActor.train()
MegatronTrainRayActor._get_rollout_data()
MegatronTrainRayActor.train_actor()
```

`RayTrainGroup.async_train()` 会对每个 train actor 调：

```python
actor.train.remote(rollout_id, rollout_data_ref, external_data=...)
```

`MegatronTrainRayActor.train()` 的结构：

```python
if self.args.offload_train:
    self.wake_up()

rollout_data = self._get_rollout_data(rollout_data_ref)

if self.role == "critic":
    result = self.train_critic(...)
else:
    self.train_actor(...)

if self.args.offload_train:
    del rollout_data
    self.sleep()
```

`_get_rollout_data()` 把 CPU/Ray 数据搬到当前 rank 的 GPU：

```python
rollout_data = process_rollout_data(...)
rollout_data["tokens"] = [
    torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device())
    for t in rollout_data["tokens"]
]
rollout_data["loss_masks"] = [
    torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device())
    for t in rollout_data["loss_masks"]
]
```

如果有 `rollout_log_probs` 或 `teacher_log_probs`，还会按 CP/qkv layout 切到当前 rank 需要的 response 片段：

```python
slice_log_prob_with_cp(log_prob, total_length, response_length, ...)
```

### actor train 内部的 RL 计算顺序

代码位置：

```text
/workspace/vime/vime/backends/megatron_utils/actor.py
MegatronTrainRayActor.train_actor()
```

主流程：

```python
data_iterator = get_data_iterator(rollout_data)
num_microbatches = rollout_data["num_microbatches"]
global_batch_sizes = rollout_data["global_batch_sizes"]

if self.args.compute_advantages_and_returns:
    # 可选：ref / teacher / old_actor logprob
    rollout_data.update(self.compute_log_prob(...))

    # critic values or external values
    ...

    compute_advantages_and_returns(self.args, rollout_data)

log_rollout_data(...)

train(
    rollout_id,
    self.model,
    self.optimizer,
    self.opt_param_scheduler,
    data_iterator,
    num_microbatches,
    global_batch_sizes,
)

self.weights_backuper.backup("actor")
```

对 PPO/GRPO 类训练来说，关键概念是：

- rollout 阶段拿到 sample/reward。
- train 阶段重新计算当前 actor 对这些 response token 的 logprob。
- 结合 reward/advantage 计算 policy loss。
- 反向更新 Megatron actor。
- 更新后的 actor 权重再同步回 vLLM，供下一轮 rollout 使用。

本轮 `kl_loss_coef=0`、`entropy_coef=0`、`VIME_SKIP_ZERO_ENTROPY_METRIC=1`，所以主关注点变成 actor policy loss 所需的 selected-token logprob，这正是 `linear_logp` 的位置。

### Megatron pipeline train step 与 loss_function

代码位置：

```text
/workspace/vime/vime/backends/megatron_utils/model.py
train()
train_one_step()
```

`train()` 会按 rollout 内的 step 切分调用 `train_one_step()`。`train_one_step()` 定义了给 Megatron pipeline engine 的 `forward_step()`：

```python
def forward_step(data_iterator, model, return_schedule_plan=False):
    batch = get_batch(...)

    linear_logp_context = None
    if _train_should_return_hidden_for_linear_logp(args, return_schedule_plan=return_schedule_plan):
        linear_logp_context = get_linear_logp_context_from_model(args, model)

    with _probe_baseline_output_layer_forward(args, model):
        with return_hidden_states_for_linear_logp(args, model, linear_logp_context):
            output_tensor = model(**forward_kwargs)

    return output_tensor, partial(
        loss_function,
        args,
        batch,
        num_microbatches,
        step_global_batch_size,
        rl_kernel_linear_logp_context=linear_logp_context,
    )
```

这个函数是 baseline/candidate 分叉的核心：

- baseline：`linear_logp_context is None`，Megatron model 正常返回 logits，后续 loss 走 `calculate_log_probs_and_entropy(logits, tokens, ...)`。
- candidate：`linear_logp_context` 非空，`return_hidden_states_for_linear_logp()` 临时让 Megatron 返回 hidden states，loss 里调用 RL-Kernel `linear_logp`，不在 PyTorch 层物化完整 logits。

Megatron 的 forward/backward 由：

```python
forward_backward_func = get_forward_backward_func()
losses_reduced = forward_backward_func(
    forward_step_func=...,
    data_iterator=data_iterator,
    model=model,
    num_microbatches=num_microbatches,
    ...
    forward_only=False,
)
```

执行。也就是说，RL-Kernel `linear_logp` 并不是绕过 Megatron 训练；它只是替换 actor loss 中“hidden/output_layer -> selected logprob”这一段，仍然在 Megatron pipeline/DDP/optimizer 框架内参与 autograd。

### 权重同步：为什么训练后必须 update_weights

RL 训练里有两个 actor 副本：

```text
Megatron actor: 训练副本，负责 forward/backward/optimizer.step
vLLM actor: rollout 副本，负责高吞吐生成 response
```

训练后 Megatron actor 权重变了。如果不把新权重同步到 vLLM，下一轮 rollout 仍然用旧策略生成，训练就会偏离 on-policy 目标。

`train.py` 因此每轮 train 后调用：

```python
actor_model.update_weights()
```

它最终广播到每个 Megatron rank：

```python
RayTrainGroup.update_weights()
-> MegatronTrainRayActor.update_weights()
-> self.weight_updater.update_weights()
```

本轮 colocate 下 `self.weight_updater` 是：

```text
/workspace/vime/vime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
UpdateWeightFromTensor
```

### colocate 权重同步的完整数据流

代码位置：

```text
/workspace/vime/vime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
UpdateWeightFromTensor.update_weights()
UpdateWeightFromTensor._send_hf_params()
_send_to_colocated_engine()

/workspace/vime/vime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py
HfWeightIteratorDirect.get_hf_weight_chunks()

/workspace/vime/vime/backends/megatron_utils/update_weight/common.py
named_params_and_buffers()
all_gather_params_async()

/workspace/vime/vime/backends/vllm_utils/vllm_engine.py
VLLMEngine.update_weights_from_tensor()
```

整体流程：

```text
Megatron sharded params
-> collect global param metadata
-> PP/EP broadcast
-> TP all-gather full param
-> convert Megatron names/layout to HF/vLLM names/layout
-> build CUDA IPC handles
-> Gloo gather IPC payloads to vLLM slot leader
-> Ray call VLLMEngine.update_weights_from_tensor()
-> HTTP POST /update_weights to vLLM server
```

`UpdateWeightFromTensor.update_weights()` 先让 vLLM 暂停生成并清 cache：

```python
if rank == 0:
    ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
    ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
```

然后每个 colocated engine 进入 vLLM weight update mode：

```python
if self._ipc_engine is not None and rank == self._ipc_gather_src:
    ray.get(self._ipc_engine.start_weight_update.remote(is_checkpoint_format=True))
```

接着从 Megatron 取 actor 权重：

```python
megatron_local_weights = self.weights_getter()
```

本轮 `weights_getter` 来自：

```python
weights_getter=lambda: self.weights_backuper.get("actor")
```

也就是刚训练完并 `backup("actor")` 的 actor 参数。

### Megatron 参数如何变成 HF/vLLM 参数

`HfWeightIteratorDirect.get_hf_weight_chunks()` 是核心：

```python
for bucket_idx, megatron_local_param_infos in enumerate(self.megatron_local_param_info_buckets, start=1):
    megatron_full_params = _get_megatron_full_params(megatron_local_param_infos, megatron_local_weights)
    hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)
    yield hf_named_tensors
```

为什么要分 bucket：

- Qwen3-30B-A3B 参数很大。
- 一次性 all-gather + 转 HF + IPC 可能打爆显存。
- `VIME_UPDATE_WEIGHT_BUFFER_SIZE=134217728` 把每个 bucket 控制在 128MiB 量级。
- 本轮日志中的 `[weight-sync] bucket ...` 就来自这里。

`_get_megatron_full_params()` 做多级并行收集：

1. 参数只在 `info.src_rank` 上真实存在，其它 rank 创建 empty tensor。
2. 如果 PP>1，跨 pipeline parallel group broadcast。
3. 如果 EP>1，expert 参数跨 expert parallel group broadcast。
4. 恢复 tensor parallel attrs。
5. 调 `all_gather_params_async()` 跨 TP/ETP all-gather 成完整权重。

```python
gathered_params = all_gather_params_async(list(zip(megatron_local_param_infos, params, strict=False)))
```

然后转换成 HF/vLLM 命名：

```python
hf_named_tensors.extend(
    convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
)
```

本轮对 Qwen3MoE 做了两个关键修正：

1. layernorm 名字兼容：

   ```python
   rest in {"self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"}
   rest in {"mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight", "post_attention_layernorm.weight"}
   ```

2. grouped expert 权重拆成 per-expert：

   ```python
   if rest == "mlp.experts.weight1":
       expert_tensors = param.view(num_local_experts, args.hidden_size, -1).transpose(-1, -2)
       target = "linear_fc1"
   elif rest == "mlp.experts.weight2":
       expert_tensors = param.view(num_local_experts, -1, args.hidden_size).transpose(-1, -2)
       target = "linear_fc2"
   ```

原因是 Megatron 训练侧的 grouped MoE 参数布局和 vLLM/HF 推理侧的 per-expert 参数布局不同。如果这里没拆对，vLLM 能收到权重，但专家层语义会错。

### CUDA IPC 到 vLLM

HF named tensors 准备好后：

```python
refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
ray.get(refs)
```

colocate 路径进入：

```python
_send_to_colocated_engine(
    hf_named_tensors,
    ipc_engine=self._ipc_engine,
    ipc_gather_src=self._ipc_gather_src,
    ipc_gather_group=self._ipc_gather_group,
    weight_version=self.weight_version,
)
```

如果一个 vLLM engine 使用多个 GPU slot，先在 Gloo group 内 gather 每个 rank 的 IPC payload：

```python
dist.gather_object(payload, object_gather_list=gathered_payloads, dst=ipc_gather_src, group=ipc_gather_group)
```

slot leader 合并 payload 后调用 vLLM engine：

```python
ipc_engine.update_weights_from_tensor.remote(**merged, weight_version=str(weight_version))
```

`VLLMEngine.update_weights_from_tensor()` 再通过 HTTP 调 vLLM server：

```python
payload = {"names": names, "dtype_names": dtype_names, "shapes": shapes}
payload["ipc_handles_pickled"] = base64.b64encode(cloudpickle.dumps(ipc_handles)).decode("utf-8")
result = self._make_request("update_weights", {"update_info": payload})
self._weight_version = str(weight_version)
```

也就是说，真正的大 tensor 不通过 Ray object store 拷贝；Ray/HTTP 传的是 CUDA IPC handle 和 metadata，vLLM 进程打开 handle 后读取同 GPU 上的 tensor。

每个 bucket 完成后释放 IPC cache：

```python
del long_lived_tensors, hf_named_tensors
torch.cuda.ipc_collect()
```

所有 bucket 完成后退出 vLLM weight update mode：

```python
if self._ipc_engine is not None and rank == self._ipc_gather_src:
    ray.get(self._ipc_engine.finish_weight_update.remote())
```

最后恢复生成：

```python
if rank == 0:
    ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

### 分布式权重同步分支

本轮是 colocate，所以主路径是 `UpdateWeightFromTensor`。但代码还支持非 colocate：

```text
/workspace/vime/vime/backends/megatron_utils/update_weight/update_weight_from_distributed.py
UpdateWeightFromDistributed
```

这个分支的思想是：

```text
Megatron trainer rank 0 + vLLM engine GPUs
-> 建 NCCLWeightTransferEngine group
-> Ray 传 metadata
-> NCCL broadcast tensor 到远端 engine
```

`UpdateWeightFromTensor` 里也有 mixed colocate/distributed 支持：

```python
self.use_distribute = len(rollout_engines) > colocate_engine_nums
```

如果 rollout engines 有一部分不在 actor GPU 范围内，就 colocated engine 走 IPC，剩余 engine 走 distributed NCCL。

### 为什么 T03 baseline 会卡在 wake_up weights

T03 baseline 第一次失败在：

```text
/wake_up?tags=weights
ConnectionRefusedError
```

框架级解释：

1. `train.py` 在 train 后准备更新 rollout 权重。
2. colocate/offload 模式下先 `rollout_manager.onload_weights()`。
3. `RolloutServer.onload_weights()` 会对需要 offload 的 server group 调：

   ```python
   engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS])
   ```

4. `VLLMEngine.resume_memory_occupation()` HTTP POST 到 vLLM server：

   ```python
   POST /wake_up?tags=weights
   ```

5. 如果 vLLM APIServer/EngineCore 已退出，15000 端口无人监听，就会 `ConnectionRefusedError`。

所以这个错误不是 Megatron loss 或 `linear_logp` 本身失败，而是 rollout server 生命周期/残留状态问题。清理 Ray/vLLM/redis 后重跑成功，也符合这个判断。

### 框架级排错顺序

如果后续继续跑 T07 或其它配置，建议按层次排错：

1. Ray 层：
   - `ray status`
   - Ray job 是否启动。
   - placement group 是否 ready。
   - actor 是否 `DEAD` 或 `PENDING`。

2. vLLM 层：
   - `/health` 是否 200。
   - `VLLMEngine` 是否成功 spawn server process。
   - `CUDA_VISIBLE_DEVICES` 是否对应预期 GPU。
   - sleep/wake 的 `/sleep`、`/wake_up?tags=weights`、`/wake_up?tags=kv_cache` 是否成功。

3. rollout 数据层：
   - `RolloutManager.generate()` 是否返回 `Sample`。
   - `tokens/response_lengths/loss_masks/rewards` 是否长度一致。
   - `build_dp_schedule()` 是否满足 `RBS*NSP >= DP` 和 `GBS <= RBS*NSP`。

4. Megatron train 层：
   - `MegatronTrainRayActor._get_rollout_data()` 是否能把数据搬上 GPU。
   - `train_actor()` 是否能算 advantage/logprob/loss。
   - `train_one_step()` 是否有 finite loss/grad norm。

5. RL-Kernel 算子层：
   - 是否出现 `Using RL-Kernel linear_logp op: FusedLinearLogpSM90Op`。
   - 是否出现 `Using fused-tile bf16 full-gradient tensor-parallel linear_logp fast path.`。
   - `rl_kernel_fallback_count_delta` 是否为 0。

6. 权重同步层：
   - `[weight-sync] bucket ... begin/all-gather/HF convert/IPC payload/update returned` 是否连续出现。
   - vLLM `start_weight_update` / `finish_weight_update` 是否成功。
   - `weight_version` 是否随 rollout 增长。

7. 显存/offload 层：
   - rollout 生成后 vLLM 是否 offload。
   - train 前 Megatron 是否 wake。
   - train 后 vLLM 是否只先 wake weights，再 update weights，再 wake kv。
   - `actor_train_peak_reserved_delta_mb` 和 `peak_vram_gb` 是否异常上升。

这个分层视角比只看命令行更适合理解 RL 框架：vime 把“生成”和“训练”拆成两个执行系统，vLLM 负责高吞吐 rollout，Megatron 负责大模型训练，中间通过 Ray object refs 传样本、通过 CUDA IPC/NCCL 同步权重。

## 已完成结果

T01、T03、T06 都完成了 candidate 与 baseline 的完整 12 轮 no-trace 对比。T07 做过 trace 尝试和一次后续 no-trace 重启，但按用户后续要求已经停止，不纳入正式指标。

| 配置 | baseline log | candidate log |
| --- | --- | --- |
| T01 | `/workspace/logs/8gpu_T01_full_baseline_20260707_142633/ray_job_baseline.log` | `/workspace/logs/8gpu_T01_full_cuda_20260707_103211/ray_job_cuda.log` |
| T03 | `/workspace/logs/8gpu_T03_full_baseline_20260707_150229/ray_job_baseline.log` | `/workspace/logs/8gpu_T03_full_cuda_20260707_105946/ray_job_cuda.log` |
| T06 | `/workspace/logs/8gpu_T06_full_baseline_20260707_124934/ray_job_baseline.log` | `/workspace/logs/8gpu_T06_full_cuda_20260707_115418/ray_job_cuda.log` |

完整训推总览：

| 配置 | baseline status | candidate status | baseline step s | candidate step s | baseline rollout s | candidate rollout s | baseline peak reserved GB | candidate peak reserved GB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T01 | success | success | 104.18 | 105.76 | 24.34 | 24.16 | 20.60 | 19.89 |
| T03 | success | success | 155.63 | 156.55 | 74.56 | 72.68 | 24.14 | 22.49 |
| T06 | success | success | 232.20 | 228.40 | 146.76 | 143.07 | 49.26 | 46.23 |

单算子主指标：

| 配置 | tokens/call baseline | tokens/call candidate | fwd ms baseline | fwd ms candidate | fwd speedup | fwd+bwd ms baseline | fwd+bwd ms candidate | fwd+bwd speedup | reserved delta baseline | reserved delta candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T01 | 796.44 | 796.44 | 5.14 | 3.51 | 1.46x | 15.24 | 12.37 | 1.23x | 4056 MB | 3112 MB |
| T03 | 1820.44 | 1820.44 | 7.78 | 3.36 | 2.32x | 18.56 | 10.37 | 1.79x | 6684 MB | 4862 MB |
| T06 | 6769.78 | 6826.67 | 14.52 | 7.62 | 1.91x | 33.96 | 18.50 | 1.84x | 32342 MB | 26710 MB |

口径说明：

- 表中单算子指标来自 rollout 3-11 的稳定窗口。
- baseline 有偶发 native logprob spike，因此 T01/T03/T06 的 baseline 单算子主表剔除了已确认的 spike 轮次。
- T01 是 smoke 档，整体 step time candidate 没有胜出，因为 rollout、weight sync、调度和日志等固定开销占比更大；但单算子仍快。
- T03/T06 放大了每次 `linear_logp` 的 token 数，candidate 的收益开始稳定体现。
- T06 candidate 比 baseline 少约 3.03GB full-run peak reserved；单算子 memory probe 的 reserved delta 少约 5.5GB。

## 配置矩阵经验

本轮实际跑通的关键配置：

| 配置 | RBS | NSP | GBS | MAX_TOKENS | RESP | VLLM_MEM | VLLM_MAX_MODEL_LEN | 定位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T01 | 2 | 2 | 4 | 2048 | 512 | 0.40 | 2048 | smoke |
| T03 | 2 | 2 | 4 | 4096 | 1536 | 0.45 | 4096 | target-low |
| T06 | 4 | 2 | 8 | 8192 | 3072 | 0.45 | 4096 | max-token-call probe |
| T07 | 4 | 2 | 8 | 8192 | 3968 | 0.46 | 4096 | trace/interrupted，不纳入正式数据 |

经验：

- 先跑 candidate，再补 baseline。candidate 如果 OOM，baseline 大概率没有必要先跑。
- 想放大 `linear_logp` 收益，优先提高 `RESP` 和 `MAX_TOKENS_PER_GPU`，再提高 `GBS`/`NSP`。
- T06 是当前最清晰的宣传档：tokens/call 达到 6.8k，fwd+bwd 约 1.84x，完整 step 也有小幅优势。
- T07 trace 开销和 profiler 落盘行为会影响训练推进，不能拿 trace run 与 no-trace baseline 直接比较。

## baseline 算子路径

baseline 的本质是两段式：

1. Megatron output layer 先计算完整 vocab logits。
2. vime 再对完整 logits 做 selected-token logprob。

对应代码位置：

- `/workspace/Megatron-LM/megatron/core/models/gpt/gpt_model.py`：GPT model 使用 `LinearCrossEntropyModule`/output layer。
- `/workspace/vime/vime/backends/megatron_utils/model.py`：`_probe_baseline_output_layer_forward()` 在 output layer 上挂 hook，记录 forward 和 forward+backward CUDA event。
- `/workspace/vime/vime/backends/megatron_utils/loss.py`：`get_log_probs_and_entropy()` 在 baseline 路径下调用原生 logprob。
- `/workspace/vime/vime/utils/ppo_utils.py`：`calculate_log_probs_and_entropy()` 里会对 `logits.clone()` 调 `compute_log_probs()`。

baseline 关键逻辑可以简化成：

```python
# loss.py
if linear_logp_context is None:
    logits = logits.contiguous()
    log_prob_full, entropy_full = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
    )
```

`calculate_log_probs_and_entropy()` 里又会做：

```python
log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
```

对 Qwen3-30B-A3B 这类 vocab 很大的模型，这意味着 baseline 的 hot path 会显式处理 `[T, V]` 形状的 logits：

- `T` 是 packed tokens，本轮 T06 大约 6.8k tokens/call。
- `V` 是 padded vocab，本轮 TP=2 时每 rank 是本地 vocab shard，全局 vocab 仍很大。
- output layer 要先产生完整 logits。
- logprob 还要对完整 logits 做 softmax/logsumexp/gather 相关操作。
- 即使最终 PPO loss 只需要 selected token 的 logprob，baseline 仍为所有 vocab token 支付了额外 HBM 读写和中间 tensor 成本。

baseline 计时是这样拼成 `baseline_linear_logp_*` 的：

- `output_layer_*` 记录 output layer 计算完整 logits 的耗时和 CUDA event。
- `native_logprob_*` 记录 `calculate_log_probs_and_entropy()` 的耗时和 CUDA event。
- `baseline_linear_logp_*` 把二者加总，形成与 candidate `linear_logp` 可比较的总成本。

这也解释了 baseline spike 的来源：output layer 和 native logprob 是两个独立阶段，任一阶段发生调度、clone、allocator、softmax 或 TP 通信抖动，都会放大到 `baseline_linear_logp_*` 上。本轮看到 T01/T03/T06 baseline 均有 native logprob spike，因此宣传图使用稳定窗口，不使用 spike 轮次夸大结论。

## candidate 算子路径

candidate 的路径是直接用 RL-Kernel 在 hidden state 上计算 selected logprob，避免完整 logits 作为 Python/PyTorch 层面的中间结果。

对应代码位置：

- `/workspace/vime/vime/backends/megatron_utils/rl_kernel.py`
  - `_get_linear_logp_op()` 根据 `VIME_RL_KERNEL_LINEAR_LOGP_BACKEND=cuda` 加载 `FusedLinearLogpSM90Op`。
  - `get_linear_logp_context_from_model()` 从 Megatron model 取 `lm_head_weight`、bias、TP group、`vocab_start_index`、`global_vocab_size`。
  - `return_hidden_states_for_linear_logp()` 临时把 Megatron `post_process=False`，让模型返回 hidden states 而不是 logits。
  - `maybe_compute_linear_logp()` 直接调用 RL-Kernel op。
- `/workspace/RL-Kernel/rl_engine/kernels/ops/cuda/loss/linear_logp.py`
  - `FusedLinearLogpSM90Op.apply()` 选择 SM90 tensor-parallel fast path。
  - `_TensorParallelLinearLogpFusedTileBF16Function` 负责 TP 下 forward/backward。
- `/workspace/RL-Kernel/csrc/cuda/fused_linear_logp_sm90.cu`
  - `fused_linear_logp_sm90_forward_impl()` 是 fused forward CUDA 入口。
  - `fused_linear_logp_sm90_backward()` 是 fused backward CUDA 入口。

candidate 关键逻辑可以简化成：

```python
# vime/backends/megatron_utils/rl_kernel.py
op = FusedLinearLogpSM90Op()
log_prob = op(
    hidden_states,
    lm_head_weight,
    target_ids.long(),
    bias,
    tp_group=context.tp_group,
    vocab_start_index=context.vocab_start_index,
    global_vocab_size=context.global_vocab_size,
)
```

RL-Kernel 里 `FusedLinearLogpSM90Op` 的 docstring 已经说明了核心目标：

```python
Computes log_softmax(hidden @ W^T + b)[target] without materializing the [N, V] logits.
```

本轮 candidate 命中的 fast path 是日志里的：

```text
Using fused-tile bf16 full-gradient tensor-parallel linear_logp fast path.
```

命中条件包括：

- SM90/Hopper 设备。
- hidden 和 weight 是 bf16。
- TP vocab shard 场景可识别。
- bias 为 None。
- full-gradient 训练需要 hidden 或 weight 的梯度。
- `RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL=1`。

CUDA forward 的关键实现点：

- 用 TMA descriptor 读取 hidden tile 和 weight tile。
- 用 WGMMA/Tensor Core 做 tile GEMM。
- 在 tile 内累计每个 token 的局部 max、sum exp、target logit。
- forward 返回的是 `[N]` 的 selected logprob 或 target logit/lse，不返回 `[N, V]` logits。
- TP 场景下各 rank 先算 local shard 的 `local_target_logit` 和 `local_lse`，再合并成 global selected logprob。

CUDA backward 的关键实现点：

- full-gradient path 以 vocab tile 为单位重算或生成局部 `dlogits`。
- 对 `grad_hidden` 和 `grad_weight` 用 tile GEMM 累积。
- 避免 Python 层 chunk loop 和大量小 matmul dispatch。
- bf16 输入下尽量让 GEMM 输入保持 Tensor Core 友好的布局和 dtype。

## 为什么 candidate 比 baseline 好

这不是简单的“少一个 kernel launch”，主要是计算图和数据流都变了。

### 1. 避免完整 logits 物化

baseline 为 selected logprob 先生成完整 `[T, V]` logits。以 T06 约 6.8k tokens/call 为例，即使 TP=2 后每 rank 只看本地 vocab shard，这个矩阵仍然很大。完整 logits 会带来：

- output layer 写出大矩阵。
- native logprob 再读入大矩阵。
- `logits.clone()` 带来额外读写。
- softmax/logsumexp 相关中间结果继续访问同一大矩阵。
- backward 还要为 logits 梯度走一遍大矩阵。

candidate 在 CUDA kernel 内直接围绕 selected target 计算 logsumexp 和 target logit，不把完整 logits 暴露为框架层 tensor。最终需要保存的是 `[N]` 级别的 lse/logp，以及 backward 所需的少量状态。显存峰值和 HBM traffic 都下降。

### 2. 把 linear + logprob 融合成同一个算子语义

baseline 的算子边界是：

```text
hidden -> output_layer -> logits -> calculate_log_probs_and_entropy -> selected logprob
```

candidate 的算子边界是：

```text
hidden + lm_head_weight + target_ids -> selected logprob
```

这个边界变化让 CUDA 实现可以在 tile 内边算 GEMM，边维护 max/sum/target 统计量，不需要先完成整张 logits 矩阵再进入下一步。

### 3. TP vocab shard 更自然

baseline 在 vime/Megatron 原生路径里仍然围绕 logits tensor 和后续 logprob 工具函数组织逻辑。candidate 在 `LinearLogpContext` 里明确携带：

- `tp_group`
- `vocab_start_index`
- `global_vocab_size`

RL-Kernel 的 TP path 让每个 rank 只在本地 vocab shard 里做 fused tile forward，然后用 TP group 合并 `local_lse` 和 owned target logit。这个结构更贴近 vocab-parallel output layer 的真实分片。

### 4. backward 更少 Python 调度和 allocator 压力

baseline 的 backward 经过 output layer autograd 和 native logprob 的组合，框架层中间 tensor 更多。candidate 的 `fused_linear_logp_sm90_backward()` 在 CUDA/C++ 里组织 vocab tile 循环，减少 Python 层 chunk loop、小 kernel、小 matmul dispatch 和 allocator 抖动。

这也是为什么 T01 这种小 token/call 配置里，candidate 单算子仍快，但完整 step 不一定赢；一旦 T03/T06 把 token/call 放大，kernel hot path 占比上升，candidate 的收益就更明显。

### 5. 实测支持这个解释

| 配置 | tokens/call | fwd speedup | fwd+bwd speedup | full-run reserved saving |
| --- | ---: | ---: | ---: | ---: |
| T01 | 796 | 1.46x | 1.23x | 0.71GB |
| T03 | 1820 | 2.32x | 1.79x | 1.65GB |
| T06 | 6827 | 1.91x | 1.84x | 3.03GB |

随着 tokens/call 增大，candidate 避免 `[T, V]` 中间结果的收益变得更实在。T06 的 full-run peak reserved 从 49.26GB 降到 46.23GB，和单算子 memory probe 的趋势一致。

## 跑通 8 卡时做过的代码级修正

这些修改不是都直接影响 `linear_logp` 性能，但它们决定了长跑能否稳定完成。

### workspace-local cache

`scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh` 和 `scripts/run-qwen3-30B-A3B.sh` 都加入了 workspace-local cache 变量。这样 Ray worker、vLLM worker、FlashInfer 编译缓存和 torch cache 不会散落到系统目录。

### vLLM spawn 和 faulthandler

`scripts/run-qwen3-30B-A3B.sh` 和 `vime/backends/vllm_utils/vllm_engine.py` 里加入：

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn
VIME_VLLM_FAULTHANDLER=1
```

实践原因：

- vLLM engine 在 Ray worker 内启动，fork 方式更容易继承复杂 CUDA/Ray 状态。
- spawn 更干净，出错时配合 faulthandler 更容易看到 Python 栈。
- T03 baseline 第一次失败表现为 vLLM APIServer/EngineCore 退出，onload weights 请求 `/wake_up?tags=weights` 时 `ConnectionRefusedError`。清理后重跑成功，因此该失败不计入性能。

### Megatron grouped GEMM validation bypass

`vime/backends/megatron_utils/arguments.py` 加入：

```python
skip_grouped_gemm_capability_check = (
    os.environ.get("VIME_SKIP_MOE_GROUPED_GEMM_CAPABILITY_CHECK", "0") == "1"
    and getattr(args, "moe_grouped_gemm", False)
)
```

做法是 validation 时临时把 `args.moe_grouped_gemm=False`，跑完 Megatron validation 后恢复为 True。原因是当前环境里 Megatron 的 capability check 会阻止配置通过，但 runtime 实际需要保留 grouped GEMM 设置。

### Qwen3MoE layernorm 权重映射

`vime/backends/megatron_utils/megatron_to_hf/qwen3moe.py` 扩展了 layernorm 名称映射：

```python
rest in {"self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"}
rest in {"mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight", "post_attention_layernorm.weight"}
```

实践原因：不同 Megatron/MoE 代码路径导出的 layernorm key 不完全一致，如果不兼容这些名字，Megatron -> HF/vLLM 权重同步容易漏映射。

### grouped MLP expert 权重拆分

`vime/backends/megatron_utils/update_weight/common.py` 加入 `_iter_grouped_mlp_expert_weights()`，把 grouped expert 权重拆成 vLLM/HF 侧期望的 per-expert 权重名：

```python
if rest == "mlp.experts.weight1":
    expert_tensors = param.view(num_local_experts, args.hidden_size, -1).transpose(-1, -2)
    partition_dim = 0
    target = "linear_fc1"
elif rest == "mlp.experts.weight2":
    expert_tensors = param.view(num_local_experts, -1, args.hidden_size).transpose(-1, -2)
    partition_dim = 1
    target = "linear_fc2"
```

并保留 tensor parallel attrs：

```python
tensor.tensor_model_parallel = getattr(source, "tensor_model_parallel", False)
tensor.partition_dim = partition_dim
tensor.partition_stride = 1
tensor.parallel_mode = getattr(source, "parallel_mode", None)
```

实践原因：Qwen3 MoE 的训练侧 grouped expert 参数和推理侧 per-expert 参数命名/布局不一致，权重同步必须在 vime 侧做结构化转换，不能靠字符串硬凑。

### update weight 分桶和计时

`hf_weight_iterator_direct.py` 和 `update_weight_from_tensor.py` 增加了分桶计时日志，并支持：

```bash
VIME_UPDATE_WEIGHT_BUFFER_SIZE=134217728
```

实践原因：

- 8 卡长跑里 weight sync 是 step time 的固定成本来源之一。
- 分桶过大可能压显存或造成 Ray object 传输抖动。
- 分桶计时能区分“kernel 慢”和“权重同步慢”，否则容易把 full step 差异误归因到 `linear_logp`。

### Megatron-LM 本地兼容补丁

`/workspace/Megatron-LM` 当前有本地兼容补丁，主要用于让本环境的 Qwen3-MoE + packed THD + fallback attention 能跑通：

- `megatron/core/transformer/dot_product_attention.py`：给 non-TE DotProductAttention 增加 packed THD fallback。
- `megatron/core/transformer/moe/moe_layer.py`：通过 `MEGATRON_ALLOW_MOE_TP_WITHOUT_SP=1` 放过 MoE TP without SP 的训练校验。
- `megatron/core/transformer/moe/moe_utils.py`：TE import 失败时补 `te_general_gemm = None`，避免后续空引用。
- 其他文件有本地修改，提交 PR 前需要确认哪些属于本轮 vime/RL-Kernel 变更范围。

这类 Megatron 本地补丁要单独看待：它们保证环境可运行，但不应在宣传时被描述成 RL-Kernel 算子收益来源。

## trace 经验

T07 做过一次 trace run：

```bash
TRACE_MODE=all TRACE_ROLLOUTS=3 \
scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh T07 cuda
```

实际状态：

- 运行到 step 2/11。
- rollout 3 的 rollout-generate trace 已触发 `cudaProfilerStart`。
- profiler stop/落盘阶段没有及时返回。
- 后续为了先跑 T01/T03 baseline，已主动中断。
- trace 文件未形成可用产物。

结论：

- trace 会改变 timing，不能与 no-trace baseline/candidate 直接比较。
- 需要 trace 时应单独开 run，最好只抓一个明确 rollout/actor train 窗口。
- 正式指标必须使用 `TRACE_MODE=none`。

## debug 经验

### T03 baseline EngineDead

失败日志：

```text
/workspace/logs/8gpu_T03_full_baseline_20260707_145557/ray_job_baseline.log
```

症状：

- vLLM APIServer/EngineCore 退出。
- onload weights 调 `/wake_up?tags=weights` 连接 15000 端口失败。
- 抛出 `ConnectionRefusedError` / `requests.exceptions.ConnectionError`。

处理：

1. `ray stop --force`。
2. 杀掉残留 vLLM/train/redis 进程。
3. 确认 `nvidia-smi` 每卡只剩少量 context 占用。
4. 重跑 T03 baseline，成功完成 12 轮。

判断：这是启动/残留状态问题，不是 baseline 算子真实性能失败。

### baseline spike

T01/T03/T06 baseline 都观察到 native logprob 相关 spike：

- T01 baseline step 10 有 400ms+ 级 forward spike。
- T03 baseline step 10 有 390ms+ 级 forward spike。
- T06 baseline step 11 有数秒级 spike。

处理方式：

- 主表使用稳定窗口，剔除 spike 轮。
- 同时在最终矩阵中保留 spike 说明，避免选择性隐瞒。
- 对宣传图使用稳定窗口数据，不用 spike 夸大 speedup。

### candidate 低于 baseline 时怎么看

如果后续某个配置出现 candidate 完整 step 不如 baseline，不要立即判定算子失败，需要拆开看：

1. `train/rl_kernel_linear_logp_forward_backward_cuda_event_elapsed_s_delta` 是否仍优于 baseline。
2. `fallback` 是否为 0。
3. rollout time、weight sync time、vLLM wakeup、Ray object store 是否变化。
4. 是否 trace/profiler 开着。
5. 是否 tokens/call 太小，kernel hot path 占比被固定开销淹没。

T01 就是典型例子：candidate 单算子更快，但完整 step 略慢，原因更可能是小配置下固定开销占比太高，而不是 `linear_logp` 算子不行。

## 可宣传结论

可以对外使用的结论应限定在 T01/T03/T06 已完成 no-trace 数据上：

- 8xH100、Qwen3-30B-A3B、完整 12 rollout 链路中，candidate T01/T03/T06 均成功完成。
- candidate 均命中 `FusedLinearLogpSM90Op` fused-tile bf16 full-gradient tensor-parallel fast path。
- fallback=0。
- 单算子 forward 最多 2.32x，forward+backward 最多 1.84x。
- T06 单算子 forward+backward 从 33.96ms 降到 18.50ms。
- T06 full-run peak reserved 从 49.26GB 降到 46.23GB。
- T06 rollout time 从 146.76s 降到 143.07s，tokens/GPU/s 从 20.33 提到 21.05。

配套可视化：

```text
/workspace/vime/vime-RLK-single-op-performance-visualization.svg
```

这张图只使用 T01/T03/T06 的 no-trace 稳定窗口数据，展示 forward、forward+backward 和 single-op reserved memory delta 三组指标，并在底部画出 baseline 与 candidate 的算子数据流差异。

建议宣传表达：

```text
On Qwen3-30B-A3B 8xH100 full rollout training, RL-Kernel fused linear_logp
hits the SM90 tensor-parallel fast path with zero fallback and cuts the
linear_logp fwd+bwd CUDA time by up to 1.84x, while reducing peak reserved
memory by 3.03GB in the T06 full-run setting.
```

中文表达：

```text
在 Qwen3-30B-A3B 8xH100 完整训推链路中，RL-Kernel fused linear_logp
稳定命中 SM90 tensor-parallel fast path，fallback=0；T06 配置下单算子
forward+backward 从 33.96ms 降至 18.50ms，约 1.84x，加上 full-run
peak reserved 显存减少约 3.03GB。
```

## 后续工作

1. 暂停 T07，除非用户明确要求继续。
2. 如果继续 T07，先跑 no-trace candidate 完整 12 轮，再决定是否补 baseline。
3. 若 T07 OOM，回退 T08。
4. 提 PR 前确认 Megatron-LM 本地补丁是否进入本轮范围；当前用户关心的是 vime 和 RL-Kernel。
5. 最终矩阵里 T07 必须保持“中断/不计入指标”，不能写成 running。
6. 发布宣传图时注明 stable window 和 no-trace 口径。
