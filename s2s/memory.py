"""Aggressive free of GPU / host memory between pipeline stages (needed on 4 GB cards)."""
import gc


def free_torch():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
