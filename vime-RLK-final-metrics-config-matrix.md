# vime + RL-Kernel linear_logp 8 卡训推 12 轮配置矩阵

日期：2026-07-07 UTC

## 最终方案

本轮使用 Qwen3-30B-A3B 跑 8 卡完整训推 12 轮，不使用 debug rollout。关注点是完整 rollout + train 流程里的 `linear_logp` 单算子指标。

| 对比项 | baseline | candidate | 训练 scope | 目标 fast path | 结论 |
| --- | --- | --- | --- | --- | --- |
| full-gradient | native output layer + native logprob | `save-logits` / fused-tile full backward | `TRAIN_SCOPE=full` | `Using fused-tile bf16 full-gradient tensor-parallel linear_logp fast path.` | 主结果；用来搜索最大不 OOM |
| output-layer-only | native output layer + native logprob | `save-prob` | `TRAIN_SCOPE=output_layer`，`--only-train-params-name-list output_layer` | `Using save-probs bf16 output-only tensor-parallel linear_logp fast path.` | 可选补充；不作为主 OOM 边界 |

原因：

- 2 卡结果是 debug train-only 单算子，只能证明 isolated kernel 能快；8 卡需要验证完整训推 12 轮下是否仍稳定。
- 为最大体现 kernel 性能，配置优先放大训练侧 `tokens_per_call`：先提高 `RESP/MAX_TOKENS`，再提高 `GBS/NSP`，同时把 `VLLM_MEM` 控制在不抢训练显存的范围。
- 最大不 OOM 搜索先跑 candidate full-gradient，再对最大成功档补 baseline。

运行脚本：`scripts/benchmarks/run-qwen3-30B-A3B-8gpu-rlk-12rollout.sh`

详细步骤：`vime-RLK-8gpu-single-op-runbook.md`

## 当前实验进展

截至 2026-07-07 15:40 UTC：

- `RL-Kernel`：已切到 PR 211 对应分支；`vime`：已切到 PR 3 对应分支。
- 已完成 8 卡 12 轮 candidate：T01、T03、T06，均命中 `FusedLinearLogpSM90Op` fused-tile full-gradient fast path，fallback=0。
- 已完成 8 卡 12 轮 baseline：T06。T06 candidate 单算子稳定窗口显著快于 baseline，且 peak reserved 更低。
- 已完成 8 卡 12 轮 baseline：T01、T03、T06。T01/T03/T06 都使用 `TRACE_MODE=none`，用于和 no-trace candidate 对齐。
- T07 no-trace candidate 已跑过首轮并确认 fast path/fallback=0，随后按要求停止并重启为 trace run；后续 no-trace 重启也已按用户要求停止，未纳入正式指标。
- T07 trace candidate 使用 `TRACE_MODE=all TRACE_ROLLOUTS=3`，已完成 step 2/11，rollout 3 的 rollout-generate trace 已触发 `cudaProfilerStart`；因 profiler stop/落盘阶段未及时返回，已按后续测试优先级主动中断，trace 文件未落盘。
- 正式 metrics run 使用 `TRACE_MODE=none` 降低 profiler overhead；trace run 只作为诊断，不混入 baseline/candidate timing 对比。

## 固定环境

| 项 | 值 |
| --- | --- |
| GPU | 8x H100 80GB |
| 模型 | Qwen3-30B-A3B |
| checkpoint | HF: `/workspace/Qwen3-30B-A3B`; Megatron load: `/workspace/Qwen3-30B-A3B_vime_tp2_dev/`; ref: `/workspace/Qwen3-30B-A3B_torch_dist` |
| TP/PP/CP/EP | TP=2, PP=1, CP=1, EP=8 |
| rollout | full vLLM rollout，不使用 debug rollout |
| num rollout | `NUM_ROLLOUT=12` |
| prompt data | `/workspace/dapo-math-17k/dapo-math-17k.jsonl` |
| baseline timer | `VIME_BASELINE_LINEAR_LOGP_TIMER=1`, `VIME_BASELINE_CUDA_EVENT_TIMER=1` |
| candidate timer | `VIME_RL_KERNEL=1`, `VIME_RL_KERNEL_LINEAR_LOGP_BACKEND=cuda`, `VIME_RL_KERNEL_CUDA_EVENT_TIMER=1` |
| trace | `TRACE_MODE=train|rollout|all`，默认 `all`；actor train 用 CUDA profiler range，rollout 用 vLLM CUDA profiler endpoint |
| correctness | `kl_loss_coef=0`, `entropy_coef=0`, `VIME_SKIP_ZERO_ENTROPY_METRIC=1` |

## 配置矩阵

所有配置固定 `NUM_ROLLOUT=12`。8 卡 TP=2 时 DP 通常是 4，每轮至少要 `RBS*NSP >= 4`，且 `GBS <= RBS*NSP`。T05/T06/T07 是主要 kernel-heavy 档位：`RESP` 接近 4096 上限，`MAX_TOKENS` 提高到 6144/8192，让一个 `linear_logp` call 尽量覆盖更多 packed tokens。

| 配置ID | 规模 | 关键配置 | 运行策略 | 状态 |
| --- | --- | --- | --- | --- |
| T01 | smoke | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=2048; RESP=512; VLLM_MEM=0.40; VLLM_MAX_MODEL_LEN=2048` | candidate full 必须先跑通 | candidate success; baseline success |
| T02 | small | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=3072; RESP=1024; VLLM_MEM=0.42; VLLM_MAX_MODEL_LEN=3072` | T01 失败后定位用 | 待跑 |
| T03 | target-low | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=4096; RESP=1536; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` | candidate full -> baseline full | candidate success; baseline success on attempt 2 |
| T04 | target-mid | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=4096; RESP=2048; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` | T05 失败后回退定位 | 待跑 |
| T05 | kernel-heavy | `RBS=2; NSP=2; GBS=4; MAX_TOKENS=6144; RESP=3072; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` | candidate full -> baseline full if max | 待跑 |
| T06 | max-token-call probe | `RBS=4; NSP=2; GBS=8; MAX_TOKENS=8192; RESP=3072; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` | candidate full，成功后补 baseline | candidate success; baseline success |
| T07 | OOM probe | `RBS=4; NSP=2; GBS=8; MAX_TOKENS=8192; RESP=3968; VLLM_MEM=0.46; VLLM_MAX_MODEL_LEN=4096` | T06 成功且显存余量足够时跑 | candidate trace/no-trace attempts interrupted; not included in metrics |
| T08 | bisect | `RBS=4; NSP=2; GBS=8; MAX_TOKENS=8192; RESP=3584; VLLM_MEM=0.45; VLLM_MAX_MODEL_LEN=4096` | T06 成功但 T07 OOM 时跑 | 待跑 |

推荐搜索顺序：

```text
T01 -> T03 -> T06 -> T07
```

若 `T07` OOM 且 `T06` 成功，再跑 `T08`。若 `T06` OOM，回退 `T05 -> T04`。

## 训推 12 轮总览

统计口径：

- `run_status`：12 个 rollout 全部完成才算 success。
- `step_time_s`、`rollout_time_s`、`train_time_s`、`actor_train_time_s`：取 rollout 3-11 的均值，避开前 3 轮 warmup。
- `peak_vram_gb`：取全 12 轮最大值。
- `raw_reward`、`train_rollout_logprob_abs_diff`：取 rollout 3-11 均值。

| 配置ID | run_status baseline | run_status candidate | peak_vram_gb baseline | peak_vram_gb candidate | step_time_s baseline | step_time_s candidate | train_time_s baseline | train_time_s candidate | actor_train_time_s baseline | actor_train_time_s candidate | rollout_time_s baseline | rollout_time_s candidate | tokens_per_gpu_per_sec baseline | tokens_per_gpu_per_sec candidate | raw_reward baseline | raw_reward candidate | abs_diff baseline | abs_diff candidate | loss_finite_pass baseline | loss_finite_pass candidate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| T01 | success | success | 20.60 | 19.89 | 104.18 | 105.76 | 17.56 | 20.02 | 16.95 | 18.88 | 24.34 | 24.16 | 10.53 | 10.60 | 0.0000 | 0.0000 | 0.02668 | 0.02537 | pass | pass |
| T03 | success | success | 24.14 | 22.49 | 155.63 | 156.55 | 17.78 | 20.19 | 16.66 | 19.32 | 74.56 | 72.68 | 10.30 | 10.57 | 0.0278 | 0.0278 | 0.02264 | 0.02395 | pass | pass |
| T05 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| T06 | success | success | 49.26 | 46.23 | 232.20 | 228.40 | 21.84 | 21.92 | 20.54 | 21.62 | 146.76 | 143.07 | 20.33 | 21.05 | 0.1111 | 0.0972 | 0.02034 | 0.02224 | pass | pass |
| T07 |  | trace interrupted step 2/11 |  | 59.06 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | pass until interrupted |
| T08 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## full-gradient: baseline vs save-logits

统计口径：

- 主表填 rollout 3-11 均值。
- 需要看单轮 trace 时，额外列 rollout 3 或 `TRACE_ROLLOUTS` 指定的轮次。

| 配置ID | run_status baseline | run_status save-logits | tokens/call baseline | tokens/call save-logits | token_count_delta baseline | token_count_delta save-logits | baseline fwd CUDA ms | save-logits fwd CUDA ms | fwd speedup | baseline fwd+bwd CUDA ms | save-logits fwd+bwd CUDA ms | fwd+bwd speedup | baseline dispatch ms | save-logits dispatch ms | peak_alloc_delta MB baseline | peak_alloc_delta MB save-logits | peak_reserved_delta MB baseline | peak_reserved_delta MB save-logits | fallback |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T01 | success | success | 796.44 | 796.44 | 796.44 | 796.44 | 5.14* | 3.51 | 1.46x | 15.24* | 12.37 | 1.23x | 4.81* | 3.88 | 996.13 | 995.77 | 4056 | 3112 | 0 |
| T03 | success | success | 1820.44 | 1820.44 | 1820.44 | 1820.44 | 7.78* | 3.36 | 2.32x | 18.56* | 10.37 | 1.79x | 6.61* | 3.06 | 2149.72 | 2013.54 | 6684 | 4862 | 0 |
| T05 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| T06 | success | success | 6769.78 | 6826.67 | 6769.78 | 6826.67 | 14.52* | 7.62* | 1.91x | 33.96* | 18.50* | 1.84x | 8.37* | 5.34* | 7724.49 | 7529.68 | 32342 | 26710 | 0 |
| T07 |  | trace interrupted step 2/11 |  |  |  |  |  |  |  |  |  |  |  |  |  | 10046.53 until interrupted |  | 38922 until interrupted | 0 until interrupted |
| T08 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

解读：

- `save-logits` 必须命中 fused-tile full-gradient fast path，fallback=0。
- 主指标是 `fwd+bwd CUDA ms`；`dispatch ms` 只作辅助。
- 如果 candidate 能跑通而 baseline OOM，单独记录为 candidate 最大不 OOM 优势。
- `*`：baseline 单算子有偶发 native logprob spike，因此单算子主比较使用剔除 spike 的稳定窗口。T01 baseline step 10 出现 407/558 ms fwd/fwd+bwd spike，表中使用 3-9、11 稳定窗口；T01 3-11 median 为 5.25/15.22 ms。T03 baseline step 10 出现 391/567 ms spike，表中使用 3-9、11 稳定窗口；T03 3-11 median 为 8.05/18.43 ms。T06 baseline step 11 出现 4-5 秒级 spike，表中使用 3-10 稳定窗口；T06 3-11 median 为 14.75/34.31 ms。

## T06 candidate vs baseline 结论

| 指标 | T06 baseline | T06 candidate | 结论 |
| --- | ---: | ---: | --- |
| run status | success | success | 都完整完成 12 轮 |
| rollout time s, mean 3-11 | 146.76 | 143.07 | candidate 更快 |
| tokens/GPU/s, mean 3-11 | 20.33 | 21.05 | candidate +3.5% |
| step time s, mean 3-11 | 232.20 | 228.40 | candidate 更快 |
| train time s, mean 3-11 | 21.84 | 21.92 | 基本持平 |
| peak reserved GB, max 0-11 | 49.26 | 46.23 | candidate 少约 3.03 GB |
| tokens/call, mean 3-11 | 6769.78 | 6826.67 | 接近 |
| forward CUDA ms, stable 3-10 | 14.52 | 7.62 | candidate 约 1.91x 快 |
| fwd+bwd CUDA ms, stable 3-10 | 33.96 | 18.50 | candidate 约 1.84x 快 |
| dispatch ms, stable 3-10 | 8.37 | 5.34 | candidate 更快 |
| abs diff, mean 3-11 | 0.02034 | 0.02224 | 同一量级 |

## output-layer-only: baseline vs save-prob

只在 full-gradient 主结果完成后按需补测。

| 配置ID | run_status baseline | run_status save-prob | tokens/call | token_count_delta | baseline fwd CUDA ms | save-prob fwd CUDA ms | fwd speedup | baseline dispatch ms | save-prob dispatch ms | actor_train_s baseline | actor_train_s save-prob | peak_alloc_delta MB baseline | peak_alloc_delta MB save-prob | peak_reserved_delta MB baseline | peak_reserved_delta MB save-prob | abs_diff baseline | abs_diff save-prob | fallback |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TMAX |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## 运行日志

| 配置 | log |
| --- | --- |
| T01 full candidate | `/workspace/logs/8gpu_T01_full_cuda_20260707_103211/ray_job_cuda.log` |
| T01 full baseline | `/workspace/logs/8gpu_T01_full_baseline_20260707_142633/ray_job_baseline.log` |
| T03 full baseline attempt 1 failed | `/workspace/logs/8gpu_T03_full_baseline_20260707_145557/ray_job_baseline.log` |
| T03 full baseline attempt 2 success | `/workspace/logs/8gpu_T03_full_baseline_20260707_150229/ray_job_baseline.log` |
| T03 full save-logits | `/workspace/logs/8gpu_T03_full_cuda_20260707_105946/ray_job_cuda.log` |
| T06 full baseline | `/workspace/logs/8gpu_T06_full_baseline_20260707_124934/ray_job_baseline.log` |
| T06 full save-logits | `/workspace/logs/8gpu_T06_full_cuda_20260707_115418/ray_job_cuda.log` |
| T07 full save-logits | `/workspace/logs/8gpu_T07_full_cuda_20260707_135416/ray_job_cuda.log` |
| T07 full save-logits no-trace rerun interrupted | `/workspace/logs/8gpu_T07_full_cuda_20260707_154241/ray_job_cuda.log` |
| T07 trace dir | `/workspace/nsys_traces/8gpu_T07_full_cuda_20260707_135416/` |
| T08 full baseline |  |
| T08 full save-logits |  |
| TMAX output baseline |  |
| TMAX output save-prob |  |

## 后续建议

| 目标 | 建议 |
| --- | --- |
| 快速找最大不 OOM | candidate full-gradient 跑 `T01 -> T03 -> T06 -> T07` |
| 最终 baseline 对比 | 只在最大成功档和一个中等档补 baseline |
| 训推 trace | `TRACE_MODE=all TRACE_ROLLOUTS=3` 同时抓 rollout generate 和 actor train |
| 12 轮稳定性 | 以 rollout 3-11 均值为主，同时确认 12 轮无 OOM/no hang |
| output-layer-only | 主结果完成后再补 `TMAX` 一组 |
