import triton
import triton.language as tl
import torch

# Toy example: Compare tl.dot(tl.ones, v) vs tl.sum(v, 0) for a single row block.
@triton.jit
def sum_vs_dot_kernel(
    v_ptr,  # pointer to input [BLOCK_M, BLOCK_N]
    out_dot_ptr,  # pointer to output for dot version [BLOCK_N]
    out_sum_ptr,  # pointer to output for sum version [BLOCK_N]
    BLOCK_M: tl.constexpr,  # block row size
    BLOCK_N: tl.constexpr,  # block col size
):
    # Program loads the whole [BLOCK_M, BLOCK_N] block from v into v_block
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    v_block = tl.load(v_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :])  # [BLOCK_M, BLOCK_N]

    # Compute dot between a block of ones and v along the ROW dimension
    ones_block = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) + 1.0
    dot_result = tl.dot(ones_block, v_block)  # [BLOCK_N]

    # Compute sum along 0 (rows)
    sum_result = tl.sum(v_block, 0)  # [BLOCK_N]

    # Store outputs
    tl.store(out_dot_ptr + offs_n, dot_result)
    tl.store(out_sum_ptr + offs_n, sum_result)

# ---- Host code to test both

BLOCK_M = 128
BLOCK_N = 64

# Prepare a random block to operate on. Shape = [BLOCK_M, BLOCK_N]
v = torch.randn((BLOCK_M, BLOCK_N), device='cuda', dtype=torch.float32)

# Allocate outputs
out_dot = torch.empty((BLOCK_N,), device='cuda', dtype=torch.float32)
out_sum = torch.empty((BLOCK_N,), device='cuda', dtype=torch.float32)

# Launch kernel (just one block)
sum_vs_dot_kernel[1](
    v,               # v_ptr
    out_dot,         # out_dot_ptr
    out_sum,         # out_sum_ptr
    BLOCK_M, BLOCK_N
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
