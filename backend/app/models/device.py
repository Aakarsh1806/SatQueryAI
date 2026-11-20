"""Optional ML device utilities; importing this module never imports Torch."""
import gc
import os
import resource
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def select_device(torch, requested=None):
    if requested:
        if requested not in {"cuda", "mps", "cpu"}:
            raise ValueError("Device must be cuda, mps, or cpu.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(torch, device):
    if device.type == "cuda": torch.cuda.synchronize()
    elif device.type == "mps": torch.mps.synchronize()


def free_cache(torch):
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()


def memory_stats(torch, device):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = {"process_peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024)}
    if device.type == "cuda": stats["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    if device.type == "mps": stats["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
    return stats
