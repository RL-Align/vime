# Linear Logp Providers

Vime can call an optional provider for the actor's `linear_logp` operation:

```bash
--linear-logp-provider package.module.provider
--linear-logp-provider-mode strict
```

The provider receives a `LinearLogpRequest` and returns a
`LinearLogpResult` from `vime.backends.megatron_utils.linear_logp_provider`.
The request contains normalized local logits, target IDs, token-row layout,
and an optional `LinearLogpContext`. The context groups a hidden-state tensor,
a local `LinearProjection`, and a `VocabPartition`, so a fused provider can
avoid depending on a particular model class or framework package.

The provider owns log-probability math, vocabulary-parallel reduction, its
numeric contract, and backend provenance. Vime retains token-row ownership,
response slicing and redistribution, and PPO/GRPO loss composition. Token
parallelism is represented by `tensor_parallel_group`; context parallelism is
row ownership and must not be used as the vocabulary reduction group.

`auto` falls back to the native implementation only if the provider cannot be
loaded or raises `LinearLogpProviderUnavailable`. `strict` rejects that case.
Provider exceptions and invalid result shapes always fail the run. A strict
result must include non-empty `backend_id` and `contract_id` and remain
connected to autograd when the request requires gradients.

The validated strict launch contract is:

```bash
--tensor-model-parallel-size 2 \
--context-parallel-size 2 \
--rollout-top-p 1.0 \
--linear-logp-provider rl_engine.integrations.vime.linear_logp_provider.provider \
--linear-logp-provider-mode strict
```

The provider owns the TP vocabulary reduction only. It does not claim
attention or FFN train/rollout consistency; those claims require runtime
readback from both Megatron and vLLM and are reported by the RL-Kernel
validation example.

The Qwen3 TP=2, CP=2 launcher also accepts a user-facing mode:

```bash
RL_KERNEL_MODE=strict scripts/run-qwen3-8B-rlkernel-tp2-cp2.sh
```

| Mode | Effective route | Fallback behavior |
| --- | --- | --- |
| `strict` | RL-Kernel training and rollout | fails closed |
| `audit` | RL-Kernel training and rollout | records complete route evidence |
| `auto` | native operators behind installed adapters | observable native fallback |
| `off` | native training and rollout | no provider, Megatron init, or vLLM plugin injection |

`strict` and `off` use the same aligned framework settings by default so they
form the recommended post-training performance comparison. Set
`RL_KERNEL_ALIGNED=0` only when intentionally testing the unaligned contract.
