# RL-Kernel `linear_logp`

vime can optionally use RL-Kernel for the actor selected-logprob path. When
`linear_logp` is enabled, the final Megatron pipeline stage returns hidden
states and vime calls the RL-Kernel operator adapter with hidden states,
LM-head weights, target token IDs, and tensor-parallel metadata.

If the operator is disabled, unavailable, or unsupported for the current
runtime shape, vime falls back to the native Megatron output-layer plus
selected-logprob path.

## Controls

Preferred Phase 1 controls:

```bash
--rlk-fast auto
--rl-kernel-ops linear_logp
```

Use strict mode when a missing or unsupported RL-Kernel path should fail the
run instead of falling back:

```bash
--rlk-fast strict
--rl-kernel-ops linear_logp
```

Legacy aliases are still accepted and resolve into the same mode config:

```bash
--enable-rl-kernel
--rl-kernel-strict
VIME_RLK_FAST=auto|strict
VIME_RL_KERNEL=1
VIME_RL_KERNEL_STRICT=1
VIME_RL_KERNEL_OPS=linear_logp
```

`VIME_LINEAR_LOGP_MEMORY_PROBE=1` enables optional CUDA memory probes around
the operator call.

## Support Matrix

| Backend | Implementation | dtype | Hardware/backend | TP | CP | Entropy | Full-gradient |
|---|---|---|---|---|---|---|---|
| `cuda_sm90` | RL-Kernel registry op, commonly `FusedLinearLogpSM90Op` when installed | Backend-defined; intended bf16/fp32 selected-logprob contracts are reported by RL-Kernel | NVIDIA SM90/Hopper CUDA backend when provided by the installed RL-Kernel package | Supported when the selected op accepts `tp_group`, `vocab_start_index`, and `global_vocab_size` | Not supported; falls back before CP redistribution | Not supported; falls back when entropy is requested | Supported only when the selected op saves hidden/weight backward state |
| `triton` | RL-Kernel registry op, commonly `TritonLinearLogpOp` when installed | Backend-defined floating input/output contract reported by RL-Kernel | CUDA devices supported by the installed RL-Kernel Triton backend | Supported when the selected op accepts TP metadata | Not supported; falls back before CP redistribution | Not supported; falls back when entropy is requested | Backend-defined; strict/full-gradient runs should validate saved-state support |
| `registry` | `RlkRegistryOperatorAdapter.linear_logp` calls `kernel_registry.get_op("linear_logp")` | Reported by the selected RL-Kernel backend; vime returns fp32 selected logprobs | Reported by the installed RL-Kernel backend | Supported when the selected op accepts TP metadata | Not supported; falls back before CP redistribution | Not supported; falls back when entropy is requested | Supported when the selected op saves hidden/weight backward state |
| `native` | Megatron output layer plus vime `calculate_log_probs_and_entropy` | vime native logits path, fp32 logprob computation | Same as native vime/Megatron execution | Supported by native vime/Megatron logprob path | Supported by native vime/Megatron CP redistribution path | Supported by native vime/Megatron path | Supported by native autograd over materialized logits |

## Runtime Metadata

Actor train-step logs include numeric metadata that is safe for W&B and
TensorBoard:

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

The process log also prints human-readable fields:

```text
requested_backend
actual_backend
backend_id
contract_id
fallback
fallback_reason
memory_probe_enabled
```

With `VIME_LINEAR_LOGP_MEMORY_PROBE=1`, vime adds operator-window memory
metrics when CUDA memory APIs are available:

```text
train/rl_kernel_linear_logp_memory_alloc_delta_mb
train/rl_kernel_linear_logp_memory_peak_alloc_delta_mb
train/rl_kernel_linear_logp_memory_reserved_delta_mb
train/rl_kernel_linear_logp_memory_peak_reserved_delta_mb
```

These timing and memory values cover the `linear_logp` operator call only. Do
not report them as full-step speed or memory claims; full-step measurements
must include rollout, communication, optimizer, weight sync, and host
scheduling.

## Fallback Behavior

Native fallback is preserved. Unsupported cases record `fallback=1` and a
structured fallback reason, then materialize logits and compute selected
logprobs through the native vime/Megatron path unless strict mode is enabled.

Common fallback reasons include:

- the optional RL-Kernel package is unavailable;
- entropy was requested;
- CP redistribution is active;
- the selected op does not accept tensor-parallel metadata;
- the model output layer or LM-head weight is unavailable.
