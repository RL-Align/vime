import os
from argparse import Namespace

import torch

from .cuda_event_timer import CudaEventTimerQueue


_COUNTER_KEYS = (
    "output_layer_call_count",
    "output_layer_token_count",
    "output_layer_dispatch_elapsed_s",
    "output_layer_cuda_event_count",
    "output_layer_cuda_event_elapsed_s",
    "output_layer_forward_backward_cuda_event_count",
    "output_layer_forward_backward_cuda_event_elapsed_s",
    "native_logprob_call_count",
    "native_logprob_token_count",
    "native_logprob_dispatch_elapsed_s",
    "native_logprob_cuda_event_count",
    "native_logprob_cuda_event_elapsed_s",
)
_COUNTERS: dict[str, float] = dict.fromkeys(_COUNTER_KEYS, 0.0)
_LAST_SNAPSHOT: dict[str, float] = dict.fromkeys(_COUNTER_KEYS, 0.0)
_CUDA_EVENT_TIMER_QUEUE = CudaEventTimerQueue()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def baseline_linear_logp_timer_enabled(args: Namespace | None = None) -> bool:
    if not _env_flag("VIME_BASELINE_LINEAR_LOGP_TIMER"):
        return False
    if args is not None and getattr(args, "enable_rl_kernel", False):
        return False
    return True


def baseline_cuda_event_timer_enabled(args: Namespace | None = None) -> bool:
    if not baseline_linear_logp_timer_enabled(args):
        return False
    return _env_flag("VIME_BASELINE_CUDA_EVENT_TIMER")


def reset_baseline_linear_logp_runtime_counters() -> None:
    _CUDA_EVENT_TIMER_QUEUE.clear()
    for key in _COUNTER_KEYS:
        _COUNTERS[key] = 0.0
        _LAST_SNAPSHOT[key] = 0.0


def get_baseline_linear_logp_runtime_counters() -> dict[str, float]:
    _CUDA_EVENT_TIMER_QUEUE.flush_ready()
    return dict(_COUNTERS)


def get_baseline_linear_logp_runtime_counter_delta() -> dict[str, float]:
    current = get_baseline_linear_logp_runtime_counters()
    delta = {key: current.get(key, 0.0) - _LAST_SNAPSHOT.get(key, 0.0) for key in _COUNTER_KEYS}
    _LAST_SNAPSHOT.update(current)
    return delta


def record_baseline_output_layer_runtime(token_count: int, elapsed_s: float) -> None:
    _COUNTERS["output_layer_call_count"] += 1.0
    _COUNTERS["output_layer_token_count"] += float(token_count)
    _COUNTERS["output_layer_dispatch_elapsed_s"] += float(elapsed_s)


def record_baseline_output_layer_cuda_event_runtime(elapsed_s: float) -> None:
    _COUNTERS["output_layer_cuda_event_count"] += 1.0
    _COUNTERS["output_layer_cuda_event_elapsed_s"] += float(elapsed_s)


def record_baseline_output_layer_forward_backward_cuda_event_runtime(elapsed_s: float) -> None:
    _COUNTERS["output_layer_forward_backward_cuda_event_count"] += 1.0
    _COUNTERS["output_layer_forward_backward_cuda_event_elapsed_s"] += float(elapsed_s)


def record_baseline_native_logprob_runtime(token_count: int, elapsed_s: float) -> None:
    _COUNTERS["native_logprob_call_count"] += 1.0
    _COUNTERS["native_logprob_token_count"] += float(token_count)
    _COUNTERS["native_logprob_dispatch_elapsed_s"] += float(elapsed_s)


def record_baseline_native_logprob_cuda_event_runtime(elapsed_s: float) -> None:
    _COUNTERS["native_logprob_cuda_event_count"] += 1.0
    _COUNTERS["native_logprob_cuda_event_elapsed_s"] += float(elapsed_s)


def queue_baseline_output_layer_cuda_event(
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
) -> None:
    _CUDA_EVENT_TIMER_QUEUE.enqueue(
        start_event,
        end_event,
        record_baseline_output_layer_cuda_event_runtime,
    )


def queue_baseline_output_layer_forward_backward_cuda_event(
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
) -> None:
    _CUDA_EVENT_TIMER_QUEUE.enqueue(
        start_event,
        end_event,
        record_baseline_output_layer_forward_backward_cuda_event_runtime,
    )


def queue_baseline_native_logprob_cuda_event(
    start_event: torch.cuda.Event,
    end_event: torch.cuda.Event,
) -> None:
    _CUDA_EVENT_TIMER_QUEUE.enqueue(
        start_event,
        end_event,
        record_baseline_native_logprob_cuda_event_runtime,
    )
