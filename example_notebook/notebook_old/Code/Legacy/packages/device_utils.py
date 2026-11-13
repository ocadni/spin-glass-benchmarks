import torch


def get_best_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device(reference=None):
    if isinstance(reference, torch.Tensor):
        return reference.device
    if reference is not None:
        return torch.device(reference)
    return get_best_device()


def empty_cache(device=None):
    device = get_device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def synchronize(device=None):
    device = get_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
