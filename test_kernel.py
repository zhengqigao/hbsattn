import triton
import triton.language as tl
import torch

@triton.jit
def sum_vs_dot_kernel(
    p_ptr,
    v_ptr,
    out_dot_ptr,
    out_sum_ptr,
    stride_p_m,      
    stride_p_n,    
    stride_v_n,    
    stride_v_dim,    
    stride_out_m,    
    stride_out_dim,  
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dim = tl.arange(0, BLOCK_DIM)
    

    p_ptrs = p_ptr + offs_m[:, None] * stride_p_m + offs_n[None, :] * stride_p_n
    v_ptrs = v_ptr + offs_n[:, None] * stride_v_n + offs_dim[None, :] * stride_v_dim
    out_dot_ptrs = out_dot_ptr + offs_m[:, None] * stride_out_m + offs_dim[None, :] * stride_out_dim
    out_sum_ptrs = out_sum_ptr + offs_dim
    

    p_block = tl.load(p_ptrs)
    v_block = tl.load(v_ptrs)

    out_dot_block = tl.dot(p_block, v_block)
    out_sum_block = tl.sum(v_block, 0)

    tl.store(out_dot_ptrs, out_dot_block)
    tl.store(out_sum_ptrs, out_sum_block)

# Host code
BLOCK_M = 64
BLOCK_N = 32
BLOCK_DIM = 16

p = torch.ones((BLOCK_M, BLOCK_N), device='cuda', dtype=torch.float32)
v = torch.randn((BLOCK_N, BLOCK_DIM), device='cuda', dtype=torch.float32)

out_dot = torch.empty((BLOCK_M, BLOCK_DIM), device='cuda', dtype=torch.float32)
out_sum = torch.empty((BLOCK_DIM,), device='cuda', dtype=torch.float32)

grid = (1,)
sum_vs_dot_kernel[grid](
    p,
    v,
    out_dot,
    out_sum,
    p.stride(0), p.stride(1),     
    v.stride(0), v.stride(1),      
    out_dot.stride(0), out_dot.stride(1),  
    BLOCK_M, BLOCK_N, BLOCK_DIM
)


dot_result = out_dot.cpu()
expected_dot = torch.matmul(p, v).cpu()
print("Triton dot result (first row):", dot_result[0])
print("Expected dot result (first row):", expected_dot[0])
print("Match dot?", torch.allclose(dot_result, expected_dot, atol=1e-5))


expected_sum = v.sum(dim=0).cpu()
sum_result = out_sum.cpu()
print("\nTriton sum result:", sum_result)
print("Expected sum result:", expected_sum)
print("Match sum?", torch.allclose(sum_result, expected_sum, atol=1e-5))