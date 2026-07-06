from vime.backends.megatron_utils.cuda_event_timer import CudaEventTimerQueue


class _FakeEvent:
    def __init__(self, *, elapsed_ms: float = 0.0, ready: bool = True) -> None:
        self._elapsed_ms = float(elapsed_ms)
        self.ready = ready

    def query(self) -> bool:
        return self.ready

    def elapsed_time(self, other) -> float:
        return float(other._elapsed_ms)


def test_cuda_event_timer_queue_flushes_when_event_becomes_ready():
    queue = CudaEventTimerQueue()
    observed = []
    start_event = _FakeEvent()
    end_event = _FakeEvent(elapsed_ms=12.5, ready=False)

    queue.enqueue(start_event, end_event, observed.append)
    assert observed == []

    end_event.ready = True
    queue.flush_ready()

    assert observed == [0.0125]


def test_cuda_event_timer_queue_clear_discards_pending_events():
    queue = CudaEventTimerQueue()
    observed = []

    queue.enqueue(_FakeEvent(), _FakeEvent(elapsed_ms=9.0, ready=False), observed.append)
    queue.clear()
    queue.flush_ready()

    assert observed == []
