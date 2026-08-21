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
