import torch
from typing import Optional, Callable, Dict, Any
import time
import numpy as np
import warnings
def _check_correctness(golden: torch.Tensor, result: torch.Tensor, name: str) -> dict:

    if golden.device != result.device:
        result = result.to(golden.device)
    
    if golden.shape != result.shape:
        # this might happen if in the format of [B,H,S,D] v.s. [S, H, D]
        # flexattention returns [B,H,S,D]
        # we will do automatica reshape
        warnings.warn(f"Shape mismatch: golden shape = {golden.shape} vs result shape = {result.shape}")
        if result.ndim == 4: 
            b, h, s, d = result.shape
            result = result.permute(0,2,1,3).reshape(-1, h, d)
        print("now result.shape:", result.shape)
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
    max_abs_idx = tuple(max_idx[0].tolist())


    # Print info for max relative error
    max_rel_val = error_info['max_rel']
    max_rel_idx = torch.nonzero(rel_error == max_rel_val)
    max_rel_idx = tuple(max_rel_idx[0].tolist())

    print(f"Mean Rel Error: {error_info['mean_rel']:.3e}")
    print(f"Mean Abs Error: {error_info['mean_abs']:.3e}")
    
    print(f"Max Rel Error: {error_info['max_rel']:.3e}, at index: {max_rel_idx}, golden: {golden[max_rel_idx].item()}, result: {result[max_rel_idx].item()}")
    print(f"Max Abs Error: {error_info['max_abs']:.3e} at index: {max_abs_idx}, golden: {golden[max_abs_idx].item()}, result: {result[max_abs_idx].item()}")

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
        'runtime_mean': mean_time,
        'runtime_std': std_time,
        'runtime_min': min_time,
        'runtime_max': max_time,
        'all_times': times.tolist(),
        'error': error_info
    }
