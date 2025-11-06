import triton
import triton.language as tl
import torch

# Toy example: Compare tl.dot(tl.ones, v) vs tl.sum(v, 0) for a single row block.
@triton.jit
def sum_vs_dot_kernel(
    p_ptr, # pointer to weight [BLOCK_M, BLOCK_N]
    v_ptr,  # pointer to input [BLOCK_N, BLOCK_DIM]
    out_dot_ptr,  # pointer to output for dot version [BLOCK_M, BLOCK_DIM]
    out_sum_ptr,  # pointer to output for sum version [BLOCK_DIM]
    BLOCK_M: tl.constexpr,  # block row size
    BLOCK_N: tl.constexpr,  # block col size
    BLOCK_DIM: tl.constexpr,  # block dim size
):
    
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_dim = tl.arange(0, BLOCK_DIM)
    p_block = tl.load(p_ptr + offs_m[:, None] + offs_n[None, :])  # [BLOCK_M, BLOCK_N]
    v_block = tl.load(v_ptr + offs_n[:,None] + offs_dim[None, :])  # [BLOCK_N, BLOCK_DIM]
    out_dot_block = tl.dot(p_block, v_block)  # [BLOCK_M, BLOCK_DIM]
    out_sum_block = tl.sum(v_block, 0)  # [BLOCK_DIM]
    tl.store(out_dot_ptr + offs_m[:, None] + offs_dim[None, :], out_dot_block)
    tl.store(out_sum_ptr + offs_dim, out_sum_block)

# ---- Host code to test both

BLOCK_M = 128
BLOCK_N = 64
BLOCK_DIM = 16

# Prepare a random block to operate on. Shape = [BLOCK_M, BLOCK_N]
v = torch.randn((BLOCK_N, BLOCK_DIM), device='cuda', dtype=torch.float32)
p = torch.ones((BLOCK_M, BLOCK_N), device='cuda', dtype=torch.float32)

# Allocate outputs
out_dot = torch.empty((BLOCK_M, BLOCK_DIM), device='cuda', dtype=torch.float32)
out_sum = torch.empty((BLOCK_M, BLOCK_DIM), device='cuda', dtype=torch.float32)
grid = (1)
# Launch kernel (just one block)
sum_vs_dot_kernel[grid](
    p,               # p_ptr
    v,               # v_ptr
    out_dot,         # out_dot_ptr
    out_sum,         # out_sum_ptr
    BLOCK_M, BLOCK_N, BLOCK_DIM
)

# Compare results on host
dot_result = out_dot.cpu().numpy()
sum_result = out_sum.cpu().numpy()
diff = (dot_result - sum_result)
print("Max abs diff:", abs(diff).max())
print("Dot result:", dot_result[:8])
print("Sum result:", sum_result[:8])
print("All close? ", torch.allclose(out_dot, out_sum))

# For manual inspection, print something if not allclose
if not torch.allclose(out_dot, out_sum):
    print("Diff:", diff[:8])
