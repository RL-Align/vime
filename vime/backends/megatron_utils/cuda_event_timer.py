from __future__ import annotations

from collections.abc import Callable

import torch


class CudaEventTimerQueue:
    """Collect CUDA event timings without synchronizing the hot path."""

    def __init__(self) -> None:
        self._pending: list[tuple[torch.cuda.Event, torch.cuda.Event, Callable[[float], None]]] = []

    def clear(self) -> None:
        self._pending.clear()

    def enqueue(
        self,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        record_runtime: Callable[[float], None],
    ) -> None:
        self._pending.append((start_event, end_event, record_runtime))
        self.flush_ready()

    def flush_ready(self) -> None:
        if not self._pending:
            return
        remaining = []
        for start_event, end_event, record_runtime in self._pending:
            if end_event.query():
                record_runtime(start_event.elapsed_time(end_event) / 1000.0)
            else:
                remaining.append((start_event, end_event, record_runtime))
        self._pending = remaining
