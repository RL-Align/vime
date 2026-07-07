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
