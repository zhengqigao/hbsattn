import torch
from typing import Optional, Callable, Dict, Any
import time
import numpy as np


def _check_correctness(golden: torch.Tensor, result: torch.Tensor, name: str) -> dict:

    if golden.device != result.device:
        result = result.to(golden.device)
    
    if golden.shape != result.shape:
        print(f"Shape mismatch: {golden.shape} vs {result.shape}")
        return None
    
    abs_error = torch.abs(golden - result)
    rel_error = abs_error / (1e-4 + torch.maximum(torch.abs(golden), torch.abs(result)))
    
    error_info = {
        'max_abs': abs_error.max().item(),
        'mean_abs': abs_error.mean().item(),
        'max_rel': rel_error.max().item(),
        'mean_rel': rel_error.mean().item(),
    }

    # Find maximum absolute error index
    max_abs_val = error_info['max_abs']
    max_idx = torch.nonzero(abs_error == max_abs_val)
    # For multi-dim, pick the first occurrence
    if max_idx.numel() > 0:
        max_idx = tuple(max_idx[0].tolist())
        print(f"[Error Plot] Max abs error ({max_abs_val:.3e}) at index: {max_idx}")
        print(f"    golden[{max_idx}] = {golden[max_idx].item()}")
        print(f"    result[{max_idx}] = {result[max_idx].item()}")
    else:
        print("[Error Plot] Could not locate max abs error index.")

    # Print info for max relative error
    max_rel_val = error_info['max_rel']
    max_rel_idx = torch.nonzero(rel_error == max_rel_val)
    if max_rel_idx.numel() > 0:
        max_rel_idx = tuple(max_rel_idx[0].tolist())
        print(f"[Error Plot] Max rel error ({max_rel_val:.3e}) at index: {max_rel_idx}")
        print(f"    golden[{max_rel_idx}] = {golden[max_rel_idx].item()}")
        print(f"    result[{max_rel_idx}] = {result[max_rel_idx].item()}")
        print(f"    abs_error[{max_rel_idx}] = {abs_error[max_rel_idx].item()}")
    else:
        print("[Error Plot] Could not locate max rel error index.")

    print(f"Max Rel Error: {error_info['max_rel']:.3e}")
    print(f"Mean Rel Error: {error_info['mean_rel']:.3e}")
    print(f"Max Abs Error: {error_info['max_abs']:.3e}")
    print(f"Mean Abs Error: {error_info['mean_abs']:.3e}")
    return error_info


def benchmark(
    benchmark_config: Dict[str, Any],
    func: Callable,
    *args,
    **kwargs
) -> dict:
    
    golden = benchmark_config.get('golden', None)
    if golden is not None:
        assert torch.isnan(golden).any() == False, "golden is not None, but it contains NaN"
    
    n_runs = benchmark_config.get('n_runs', 1)
    n_warmup = benchmark_config.get('n_warmup', 0)
    name = benchmark_config.get('name', func.__name__)
 
    print(f"🚀 ===== Benchmarking: {name} =====")

   
    times = []
    result = None
    error_info = None

    for i in range(n_runs):

        torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        
        torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        
        if golden is not None and i == 0:
            error_info = _check_correctness(golden, result, name)

        if i >= n_warmup:
            times.append(elapsed)
    
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times) 
    min_time = np.min(times) 
    max_time = np.max(times) 
    
    print(f"Average Time ({n_runs} runs w/ {n_warmup} warmups): {mean_time:.3e} ± {std_time:.3e} seconds")

    
    return {
        'result': result,
        'time_ms': mean_time,
        'time_std': std_time,
        'time_min': min_time,
        'time_max': max_time,
        'all_times': times.tolist(),
        'error': error_info
    }
