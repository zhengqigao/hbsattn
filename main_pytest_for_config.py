    # Pytest-based parameterized test for nhead_q % nhead_k == 0, causality, and softmax_scale

import pytest
import torch

from hbsattn.reference import (
    hbsattn_reference_v1_base,
    hbsattn_reference_v2_with_pytorch,
    hbsattn_reference_v3_qkallfirst,
)
from hbsattn.utils import calculate_blocks

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("nhead_q,nhead_k", [(2, 2), (4, 2), (8, 2)])  # nhead_q % nhead_k == 0
@pytest.mark.parametrize("softmax_scale", [None, 0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_attention_configs(causal, nhead_q, nhead_k, softmax_scale, dtype):
    device = torch.cuda.current_device()

    cu_k_seqlens = torch.tensor([0, 32, 61, 100, 134, 157], dtype=torch.int32, device=device)
    k_seqlen = cu_k_seqlens[-1].item()
    cu_q_seqlens = torch.tensor([0, 32, 61, 100, 134, 157], dtype=torch.int32, device=device)
    q_seqlen = cu_q_seqlens[-1].item()
    
    headdim = 16
    q_block_size = 16
    k_block_size = 16

    q = torch.randn(q_seqlen, nhead_q, headdim, device=device, dtype=dtype)
    k = torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)
    v = torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)

    num_q_block, cu_q_block, q_block_to_batch = calculate_blocks(cu_q_seqlens, q_block_size)
    num_k_block, cu_k_block, k_block_to_batch = calculate_blocks(cu_k_seqlens, k_block_size)

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

    golden_ref_v1 = hbsattn_reference_v1_base(
        q, k, v,
        cu_q_seqlens, cu_k_seqlens, block_mask,
        q_block_size, k_block_size,
        causal, softmax_scale,
        num_q_block, cu_q_block, q_block_to_batch,
        num_k_block, cu_k_block, k_block_to_batch
    )
    golden_ref_v2 = hbsattn_reference_v2_with_pytorch(
        q, k, v,
        cu_q_seqlens, cu_k_seqlens, block_mask,
        q_block_size, k_block_size,
        causal, softmax_scale,
        num_q_block, cu_q_block, q_block_to_batch,
        num_k_block, cu_k_block, k_block_to_batch
    )
    golden_ref_v3 = hbsattn_reference_v3_qkallfirst(
        q, k, v,
        cu_q_seqlens, cu_k_seqlens, block_mask,
        q_block_size, k_block_size,
        causal, softmax_scale,
        num_q_block, cu_q_block, q_block_to_batch,
        num_k_block, cu_k_block, k_block_to_batch
    )
    # Check that all golden refs are finite and close
    assert torch.all(torch.isfinite(golden_ref_v1))
    assert torch.all(torch.isfinite(golden_ref_v2))
    assert torch.all(torch.isfinite(golden_ref_v3))
    assert torch.allclose(golden_ref_v1, golden_ref_v2, atol=1e-4, rtol=1e-3)
    assert torch.allclose(golden_ref_v1, golden_ref_v3, atol=1e-4, rtol=1e-3)