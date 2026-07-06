import gc
import logging

import psutil
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_PEAK_MEMORY_CONTEXTS: dict[str, dict[str, int]] = {}


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    vm = psutil.virtual_memory()
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
        "host_total_GB": _byte_to_gb(vm.total),
        "host_available_GB": _byte_to_gb(vm.available),
        "host_used_GB": _byte_to_gb(vm.used),
        "host_free_GB": _byte_to_gb(vm.free),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info


def reset_peak_memory_tracker(name: str, device=None):
    device = torch.cuda.current_device() if device is None else device
    alloc_before = int(torch.cuda.memory_allocated(device))
    reserved_before = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)
    _PEAK_MEMORY_CONTEXTS[name] = {
        "alloc_before": alloc_before,
        "reserved_before": reserved_before,
        "peak_alloc": alloc_before,
        "peak_reserved": reserved_before,
    }


def update_peak_memory_tracker(name: str, *, peak_alloc=None, peak_reserved=None, device=None):
    device = torch.cuda.current_device() if device is None else device
    if peak_alloc is None:
        peak_alloc = torch.cuda.max_memory_allocated(device)
    if peak_reserved is None:
        peak_reserved = torch.cuda.max_memory_reserved(device)
    state = _PEAK_MEMORY_CONTEXTS.setdefault(
        name,
        {
            "alloc_before": int(torch.cuda.memory_allocated(device)),
            "reserved_before": int(torch.cuda.memory_reserved(device)),
            "peak_alloc": int(torch.cuda.memory_allocated(device)),
            "peak_reserved": int(torch.cuda.memory_reserved(device)),
        },
    )
    state["peak_alloc"] = max(state["peak_alloc"], int(peak_alloc))
    state["peak_reserved"] = max(state["peak_reserved"], int(peak_reserved))


def get_peak_memory_tracker(name: str, device=None):
    device = torch.cuda.current_device() if device is None else device
    state = _PEAK_MEMORY_CONTEXTS.get(name)
    if state is None:
        alloc_before = int(torch.cuda.memory_allocated(device))
        reserved_before = int(torch.cuda.memory_reserved(device))
        peak_alloc = max(alloc_before, int(torch.cuda.max_memory_allocated(device)))
        peak_reserved = max(reserved_before, int(torch.cuda.max_memory_reserved(device)))
    else:
        alloc_before = state["alloc_before"]
        reserved_before = state["reserved_before"]
        peak_alloc = max(state["peak_alloc"], int(torch.cuda.max_memory_allocated(device)))
        peak_reserved = max(state["peak_reserved"], int(torch.cuda.max_memory_reserved(device)))
    alloc_after = int(torch.cuda.memory_allocated(device))
    reserved_after = int(torch.cuda.memory_reserved(device))
    peak_alloc = max(peak_alloc, alloc_after)
    peak_reserved = max(peak_reserved, reserved_after)
    return {
        "alloc_before": alloc_before,
        "reserved_before": reserved_before,
        "alloc_after": alloc_after,
        "reserved_after": reserved_after,
        "peak_alloc": peak_alloc,
        "peak_reserved": peak_reserved,
    }
