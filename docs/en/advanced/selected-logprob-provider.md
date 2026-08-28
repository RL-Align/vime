# Selected-Logprob Providers

Vime can call an optional provider for the actor's selected-logprob operation:

```bash
--selected-logprob-provider package.module.provider
--selected-logprob-provider-mode strict
```

The provider is invoked after Vime has built local target rows from the
Megatron CP layout and applied rollout temperature. It receives
`SelectedLogprobRequest` and returns `SelectedLogprobResult` from
`vime.backends.megatron_utils.selected_logprob_provider`.

The provider owns selected-logprob math, vocabulary-parallel reduction, its
numeric contract, and backend provenance. Vime retains CP row ownership,
response slicing and redistribution, and PPO/GRPO loss composition. CP must
not be used as a vocabulary log-sum-exp reduction group.

`auto` falls back to the native implementation only if the provider cannot be
loaded or raises `SelectedLogprobProviderUnavailable`. `strict` rejects that
case. Provider exceptions and invalid result shapes always fail the run. A
strict provider result must include non-empty `backend_id` and `contract_id`
and remain connected to autograd when logits require gradients.

For RL-Kernel, `selected-logprob provider` is the Vime interface name. The
operator behind that interface is strict `linear_logp`, exposed at
`rl_engine.integrations.vime.linear_logp.provider`. The older
`rl_engine.integrations.vime.logp.provider` import remains compatible.

The validated strict launch contract is:

```bash
--tensor-model-parallel-size 2 \
--context-parallel-size 2 \
--rollout-top-p 1.0 \
--selected-logprob-provider rl_engine.integrations.vime.linear_logp.provider \
--selected-logprob-provider-mode strict
```

The provider owns the TP vocabulary reduction only. Vime continues to own CP
token-row layout, response extraction, and PPO/GRPO loss composition. The
provider does not claim attention or FFN train/rollout consistency; those
claims require runtime readback from both Megatron and vLLM and are reported by
the RL-Kernel validation example.

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
