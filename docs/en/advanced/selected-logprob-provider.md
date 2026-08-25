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

For the RL-Kernel WS2 provider, the validated launch contract is:

```bash
--tensor-model-parallel-size 2 \
--context-parallel-size 2 \
--rollout-top-p 1.0 \
--selected-logprob-provider rl_engine.integrations.vime.logp.provider \
--selected-logprob-provider-mode strict
```

The provider owns the TP vocabulary reduction only. Vime continues to own CP
token-row layout, response extraction, and PPO/GRPO loss composition. The
provider does not claim attention or FFN train/rollout consistency; those
claims require runtime readback from both Megatron and vLLM and are reported by
the RL-Kernel validation example.
