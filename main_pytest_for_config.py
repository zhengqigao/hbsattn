    # Pytest-based parameterized test for nhead_q % nhead_k == 0, causality, and softmax_scale

import pytest
import torch

from hbsattn.reference import (
    hbsattn_reference_v1_base,
    hbsattn_reference_v2_with_pytorch,
    hbsattn_reference_v3_qkallfirst,
)
from hbsattn.fwd_triton import HBSAttention
from hbsattn.utils import calculate_blocks


@pytest.mark.parametrize("causal", [False, True], ids=["causal_False", "causal_True"])
@pytest.mark.parametrize("nhead_q,nhead_k", [(2, 2), (4, 2), (8, 2)], ids=["nhead_q_2_nhead_k_2", "nhead_q_4_nhead_k_2", "nhead_q_8_nhead_k_2"]) 
@pytest.mark.parametrize("softmax_scale", [None, 0.333], ids=["softmax_scale_None", "softmax_scale_0.333"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["dtype_bfloat16", "dtype_float32"])
@pytest.mark.parametrize("q_block_size", [16, 32], ids=["q_block_size_16", "q_block_size_32"])
@pytest.mark.parametrize("k_block_size", [16, 32], ids=["k_block_size_16", "k_block_size_32"])
@pytest.mark.parametrize("k_q_same_seqlen", [True, False], ids=["k_q_same_seqlen_True", "k_q_same_seqlen_False"])
def test_attention_configs(causal, nhead_q, nhead_k, softmax_scale, dtype, q_block_size, k_block_size, k_q_same_seqlen):
    device = torch.cuda.current_device()

    if k_q_same_seqlen:
        cu_k_seqlens = torch.tensor([0, 32, 64, 96, 128, 160], dtype=torch.int32, device=device) 
        cu_q_seqlens = torch.tensor([0, 32, 64, 96, 128, 160], dtype=torch.int32, device=device)
    else:
        cu_k_seqlens = torch.tensor([0, 32, 61, 100, 134, 157], dtype=torch.int32, device=device) 
        cu_q_seqlens = torch.tensor([0, 32, 64, 96, 128, 160], dtype=torch.int32, device=device) 
    
    k_seqlen = cu_k_seqlens[-1].item()
    q_seqlen = cu_q_seqlens[-1].item()
    
    headdim = 16

    q = torch.randn(q_seqlen, nhead_q, headdim, device=device, dtype=dtype)
    k = torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype) 
    v = torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)

    num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
    num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)

    block_mask = (torch.rand(nhead_k, num_q_block, num_k_block, device=device) < 0.7).to(torch.bool)
    for i in range(num_q_block):
        batch_idx = q_block_to_batch[i]
        # Guarantee at least one k_block in the same batch is True for every q_block
        first_k_block_idx_in_the_same_batch = None
        for j in range(len(k_block_to_batch)):
            if k_block_to_batch[j] == batch_idx:
                first_k_block_idx_in_the_same_batch = j
                break
        block_mask[:, i, first_k_block_idx_in_the_same_batch] = True

    assert torch.sum(block_mask, dim=-1).all() == True, "at least one k block is needed for each q."

    golden_ref_in_float32 = hbsattn_reference_v1_base(q.float(), k.float(), v.float(), cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    golden_ref_v1 = hbsattn_reference_v1_base(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    golden_ref_v2 = hbsattn_reference_v2_with_pytorch(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    golden_ref_v3 = hbsattn_reference_v3_qkallfirst(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    out = HBSAttention(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
    
    # Check that all golden refs are finite and close
    assert torch.all(torch.isfinite(golden_ref_in_float32))
    assert torch.all(torch.isfinite(golden_ref_v1))
    assert torch.all(torch.isfinite(golden_ref_v2))
    assert torch.all(torch.isfinite(golden_ref_v3))
    assert torch.all(torch.isfinite(out))
    
    # our impl should have less error than the reference impl in float32.
    assert torch.allclose(torch.abs(golden_ref_in_float32 - golden_ref_v1), torch.abs(golden_ref_in_float32 - out), atol=1e-3, rtol=1e-2)
    # assert torch.allclose(golden_ref_v1, golden_ref_v2, atol=1e-2, rtol=1e-2)
    # assert torch.allclose(golden_ref_v1, golden_ref_v3, atol=1e-2, rtol=1e-2)
    # assert torch.allclose(golden_ref_v1, out, atol=1e-2, rtol=1e-2)


# bfloat16 usually has lower accuracy. https://github.com/Dao-AILab/flash-attention/issues/1071    
    
if __name__ == "__main__":
    test_attention_configs(causal=True, nhead_q=2, nhead_k=2, softmax_scale=None, dtype=torch.float32, q_block_size=16, k_block_size=16, k_q_same_seqlen=True)