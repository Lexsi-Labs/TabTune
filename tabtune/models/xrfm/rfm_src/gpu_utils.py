import math
import os
import time
from functools import wraps
from typing import Optional, Tuple

import torch

BASE_GPU_MEMORY_BYTES = 40 * 1024**3  # 40GB reference card


def with_env_var(var_name, value):
    """
    Decorator to set an environment variable for the duration of a function call.
    
    Args:
        var_name (str): The name of the environment variable to set.
        value (str): The value to set the environment variable to.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_value = os.environ.get(var_name)
            os.environ[var_name] = value
            try:
                return func(*args, **kwargs)
            finally:
                if original_value is None:
                    del os.environ[var_name]
                else:
                    os.environ[var_name] = original_value
        return wrapper
    return decorator


def get_gpu_memory_bytes(device=None) -> Tuple[Optional[int], Optional[int]]:
    """
    Return available and total GPU memory (in bytes) for the requested device.
    """
    if not torch.cuda.is_available():
        return None, None

    if device is None:
        device = torch.cuda.current_device()

    device = torch.device(device)
    if device.type != 'cuda':
        return None, None

    torch.cuda.empty_cache()
    with torch.cuda.device(device):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except RuntimeError:
            total_bytes = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            free_bytes = total_bytes - max(allocated, reserved)

    # guard against negative or zero values
    free_bytes = max(int(free_bytes), 0)
    total_bytes = max(int(total_bytes), 0)
    return free_bytes, total_bytes


def memory_scaling_factor(device=None, *, quadratic=False, base_memory_bytes=BASE_GPU_MEMORY_BYTES) -> float:
    """
    Compute a scaling factor relative to a 40GB GPU.
    
    Parameters
    ----------
    device : Union[str, torch.device, int], optional
        Target CUDA device. Defaults to the current device.
    quadratic : bool, default=False
        If True, return sqrt(memory_ratio) to account for quadratic scaling.
    base_memory_bytes : int, default=40GB
        Reference memory used for ratio calculation.
    """
    if base_memory_bytes <= 0:
        return 1.0

    free_bytes, _ = get_gpu_memory_bytes(device)
    if free_bytes is None or free_bytes == 0:
        return 1.0

    memory_ratio = max(free_bytes / base_memory_bytes, 1e-3)
    if quadratic:
        return math.sqrt(memory_ratio)
    return min(memory_ratio, 1) # never use more than 40GB VRAM setting


def resolve_device(device=None, *, default_device=None) -> torch.device:
    """
    Normalize potential device inputs to a torch.device instance.
    """
    if isinstance(device, torch.Tensor):
        device = device.device
    if isinstance(device, torch.device):
        return device
    if device is None:
        base = default_device
        if isinstance(base, torch.Tensor):
            base = base.device
        if isinstance(base, torch.device):
            return base
        if base is not None:
            return torch.device(base)
        return torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return torch.device(device)


def timer_start(device=None, *, default_device=None) -> float:
    """
    Synchronize the provided device before starting a timed section.
    """
    resolved_device = resolve_device(device, default_device=default_device)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    return time.perf_counter()


def timer_end(device, start_time: float, label: str, *, default_device=None, logger=print) -> float:
    """
    Synchronize the provided device, log elapsed time, and return the value.
    """
    resolved_device = resolve_device(device, default_device=default_device)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    if logger is not None:
        logger(f"{label}: {elapsed:.6f} seconds")
    return elapsed


def _bytes_to_gb_str(num_bytes: int) -> str:
    gb = num_bytes / (1024**3)
    return f"{gb:.2f} GB"


def print_vram(prefix: str = "", device=None) -> None:
    """
    Print a concise line with current CUDA VRAM usage: free/total and allocated/reserved.
    Controlled by env XRFM_VRAM_LOG=1 to reduce noise by default.
    """
    if not torch.cuda.is_available():
        print(f"{prefix} [VRAM] CUDA not available")
        return

    if device is None:
        device = torch.cuda.current_device()

    device = torch.device(device)
    if device.type != 'cuda':
        print(f"{prefix} [VRAM] device={device} is not CUDA")
        return

    with torch.cuda.device(device):
        try:
            free_b, total_b = torch.cuda.mem_get_info()
        except RuntimeError:
            total_b = torch.cuda.get_device_properties(device).total_memory
            allocated_b = torch.cuda.memory_allocated()
            reserved_b = torch.cuda.memory_reserved()
            free_b = total_b - max(allocated_b, reserved_b)
        allocated_b = torch.cuda.memory_allocated()
        reserved_b = torch.cuda.memory_reserved()

    name = torch.cuda.get_device_name(device)
    msg = (
        f"{prefix} [VRAM] dev={device.index} {name} | "
        f"free/total={_bytes_to_gb_str(free_b)}/{_bytes_to_gb_str(total_b)} | "
        f"alloc/res={_bytes_to_gb_str(allocated_b)}/{_bytes_to_gb_str(reserved_b)}"
    )
    print(msg)
