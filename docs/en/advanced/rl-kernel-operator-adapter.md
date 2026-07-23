# RL-Kernel Operator Adapter

vime integrates RL-Kernel as an optional operator backend through one vime-owned
boundary:

```python
from vime.backends.rl_kernel_utils import (
    RlkOperatorAdapter,
    RlkPolicyContext,
    RlkLinearLogpRequest,
    build_rlk_operator_adapter,
)
```

Production training, rollout, and audit code should depend on the
`RlkOperatorAdapter` protocol, request dataclasses, and `RlkOperatorResult`.
They should not import RL-Kernel internals directly.  The concrete adapter hides
RL-Kernel registry loading behind a lazy backend so native vime imports and
disabled modes do not require RL-Kernel to be installed.

## Ownership Boundary

vime owns:

- resolving user config and environment aliases into `RlkPolicyContext`;
- selecting whether an operator is enabled;
- passing tensors plus already-known policy, dtype, backend, and parallel
  metadata into the adapter;
- using `handled=False` results to continue through the native vime path;
- logging or aggregating structured decisions and provenance.

RL-Kernel owns:

- concrete operator implementations;
- backend capability descriptors;
- numeric contracts, including dtype tolerances and reduction semantics;
- backend-specific build/runtime fingerprints.

The adapter does not own rollout scheduling, training loops, data buffers,
weight synchronization, child process supervision, benchmark runners, or attempt
artifact layout.

## Supported Hooks

The Phase 1 boundary exposes these operator-scoped hooks:

- `query_capabilities(operator=None)`;
- `selected_logprob(RlkSelectedLogprobRequest(...))`;
- `reference_score(RlkReferenceScoreRequest(...))`;
- `linear_logp(RlkLinearLogpRequest(...))`.

`linear_logp` is the Phase 2 extension point for replacing the native
LM-head-plus-selected-logprob path without materializing full logits.

## Testing

Use `NoopRlkOperatorAdapter` for disabled/default behavior and
`MockRlkOperatorAdapter` for unit tests that need successful or unsupported
operator behavior without importing RL-Kernel.  Boundary tests also scan vime
production modules to ensure RL-Kernel execution internals stay outside vime
code paths.
