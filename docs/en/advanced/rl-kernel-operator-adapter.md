# RL-Kernel Operator Adapter

`vime.backends.rl_kernel_utils` is the vime-owned boundary for optional RL-Kernel-backed operator calls. Production vime code should depend on `RlkOperatorAdapter`, `NoOpRlkOperatorAdapter`, or the input dataclasses from this package, not on RL-Kernel internals.

The boundary has three jobs:

- carry resolved vime policy into the adapter (`fast`, `consistency`, and `enabled_ops`);
- map borrowed vime runtime tensors and metadata into operator-call inputs without taking ownership of rollout, training, Ray, Megatron, or artifact lifecycles;
- isolate the only optional RL-Kernel registry touchpoint, `rl_engine.kernels.registry.kernel_registry.get_op(...)`.

When RL-Kernel modes are disabled, `build_rlk_operator_adapter(...)` returns `NoOpRlkOperatorAdapter`. The no-op adapter reports disabled decisions and lets the native vime path continue unchanged.

For tests, use `MockRlkOperatorAdapter`. It implements the same protocol without importing RL-Kernel, CUDA, Triton, or distributed runtime packages.

The current Phase 1 hooks are:

- `capability(op_name)`;
- `contract(op_name, runtime=..., metadata=...)`;
- `selected_logprobs(SelectedLogprobInputs)`;
- `reference_logprobs(ReferenceScoreInputs)`;
- `linear_logp(LinearLogpInputs)`;
- `provenance()`.

Future `linear_logp` integration work should extend this adapter package rather than adding direct `rl_engine` imports to Megatron, rollout, or training modules.
