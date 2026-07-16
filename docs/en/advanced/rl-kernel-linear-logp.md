# RL-Kernel `linear_logp`

vime can optionally use RL-Kernel for the actor selected-logprob path. When `linear_logp` is enabled, Megatron returns hidden states on the final pipeline stage and vime calls RL-Kernel with hidden states, LM-head weights, target token IDs, and tensor-parallel metadata. If the backend is unavailable or the runtime contract is unsupported, vime falls back to the native Megatron output-layer plus selected-logprob path.

## Controls

```bash
--enable-rl-kernel
--rl-kernel-ops linear_logp
```

Environment aliases:

```bash
VIME_RL_KERNEL=1
VIME_RL_KERNEL_OPS=linear_logp
VIME_RL_KERNEL_LINEAR_LOGP_BACKEND=auto|registry|cuda|sm90|cuda_sm90|triton
VIME_RL_KERNEL_STRICT=1
```

`VIME_RL_KERNEL_STRICT=1` turns an unsupported RL-Kernel path into an error instead of a native fallback.

## Support Matrix

| Backend selector | Implementation | dtype | Hardware/backend | TP | CP | Entropy | Full-gradient |
|---|---|---|---|---|---|---|---|
| `cuda`, `sm90`, `cuda_sm90` | `FusedLinearLogpSM90Op` | bf16 inputs, fp32 selected logprob output | NVIDIA SM90/Hopper CUDA build with RL-Kernel extension | Supported through `tp_group`, `vocab_start_index`, `global_vocab_size` | Not supported; falls back before CP redistribution | Not supported; falls back when entropy is requested | Supported when the installed RL-Kernel op saves backward state |
| `triton` | `TritonLinearLogpOp` | Backend-defined floating input/output contract | CUDA devices supported by the installed Triton backend | Supported only when the op accepts TP metadata | Not supported; falls back before CP redistribution | Not supported; falls back when entropy is requested | Backend-defined; strict/full-gradient runs should validate saved-state support |
| `auto`, `registry` | `kernel_registry.get_op("linear_logp")` | Reported by the selected RL-Kernel backend | Reported by the selected RL-Kernel backend | Supported only when the selected op accepts TP metadata | Not supported; falls back before CP redistribution | Not supported; falls back when entropy is requested | Reported by the selected RL-Kernel backend |
| native fallback | Megatron output layer + vime `calculate_log_probs_and_entropy` | vime native logits path, fp32 logprob computation | Same as native vime/Megatron execution | Supported by the native vime/Megatron logprob path | Supported by the native vime/Megatron CP redistribution path | Supported by the native vime/Megatron path | Supported by native autograd over materialized logits |

## Runtime Metadata

Each actor train-step log includes numeric metadata that is safe for W&B and TensorBoard:

```text
train/rl_kernel_linear_logp_fallback
train/rl_kernel_linear_logp_backend_descriptor_id
train/rl_kernel_linear_logp_contract_descriptor_id
train/rl_kernel_linear_logp_fallback_reason_descriptor_id
train/rl_kernel_linear_logp_call_count_total
train/rl_kernel_linear_logp_call_count_delta
train/rl_kernel_linear_logp_token_count_total
train/rl_kernel_linear_logp_token_count_delta
train/rl_kernel_linear_logp_dispatch_elapsed_s_total
train/rl_kernel_linear_logp_dispatch_elapsed_s_delta
train/rl_kernel_linear_logp_tokens_per_call_total
train/rl_kernel_linear_logp_tokens_per_call_delta
```

The regular process log also prints the human-readable fields:

```text
requested_backend
actual_backend
backend_id
contract_id
fallback
fallback_reason
memory_probe_enabled
```

Set `VIME_LINEAR_LOGP_MEMORY_PROBE=1` to add operator-window memory probe fields:

```text
train/rl_kernel_linear_logp_memory_alloc_delta_mb
train/rl_kernel_linear_logp_memory_peak_alloc_delta_mb
train/rl_kernel_linear_logp_memory_reserved_delta_mb
train/rl_kernel_linear_logp_memory_peak_reserved_delta_mb
```

These are operator-level measurements around the `linear_logp` call. They should not be reported as full-step speed or memory claims. Full-step claims must include rollout, communication, optimizer, weight sync, and host scheduling.

## Fallback Behavior

Native fallback is preserved. Unsupported cases record `fallback=1` and a structured fallback reason, then materialize logits and compute selected logprobs through the native vime/Megatron path unless strict mode is enabled.

Common fallback reasons include:

- optional RL-Kernel package is unavailable;
- entropy was requested;
- CP redistribution is active;
- the selected op does not accept tensor-parallel metadata;
- the model output-layer or LM-head weight is unavailable.
