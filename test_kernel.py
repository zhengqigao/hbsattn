import triton 
import triton.language as tl
import torch

@triton.jit
def test_kernel(x: tl.tensor, y: tl.tensor, z: tl.tensor):
    
    return x + y + z

x = torch.randn(1024, 1024)
y = torch.randn(1024, 1024)
z = torch.randn(1024, 1024)

test_kernel[1024, 1024](x, y, z)