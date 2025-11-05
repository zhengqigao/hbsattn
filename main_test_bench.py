import torch
from hbsattn.fwd_triton import HBSAttention
from hbsattn.reference import hbsattn_reference_v1_base, hbsattn_reference_v2_with_pytorch, hbsattn_reference_v3_qkallfirst
from hbsattn.utils import calculate_blocks
from hbsattn.benchmark import benchmark



if __name__ == "__main__":

    
    
    device = torch.cuda.current_device()
    # cu_seqlens = torch.tensor([0, 4], dtype=torch.int32, device=device)
    # max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    # seqlen = cu_seqlens[-1].item()
    
    # nhead = 1
    # headdim = 1
    # q_block_size = 2
    # k_block_size = 2
    
    dtype = torch.float32
    cu_seqlens = torch.tensor([0, 32, 64, 96, 128, 160], dtype=torch.int32, device=device) # [0, 32, 64, 96, 128, 160]
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    seqlen = cu_seqlens[-1].item()
    
    nhead = 2
    headdim = 16
    q_block_size = 16
    k_block_size = 16

    q = torch.randn(seqlen, nhead, headdim, device=device, dtype=dtype)
    k = torch.randn(seqlen, nhead, headdim, device=device, dtype=dtype)
    v = torch.randn(seqlen, nhead, headdim, device=device, dtype=dtype)
    

    num_q_block, cu_q_block, q_block_to_batch = calculate_blocks(cu_seqlens, q_block_size)
    num_k_block, cu_k_block, k_block_to_batch = calculate_blocks(cu_seqlens, k_block_size)


    # Important: note if block_mask is not set properly, then it is possible for a q block not to attend to any k block, and cause the output to be NaN.
    block_mask = (torch.rand(nhead, num_q_block, num_k_block, device=device) < 0.7).to(torch.bool)
    for i in range(num_q_block):
        batch_idx = q_block_to_batch[i]
        first_k_block_idx_in_the_same_batch = None
        for j in range(len(k_block_to_batch)):
            if k_block_to_batch[j] == batch_idx:
                first_k_block_idx_in_the_same_batch = j
                break
        block_mask[:,i,first_k_block_idx_in_the_same_batch] = True # this can make sure q will attend to the first k block in the same batch.

    # block_mask = block_mask.fill_(1).contiguous()
    
    assert torch.sum(block_mask, dim=-1).all() == True, "at least one k block is needed for each q."
    
    # run once to get a golden reference
    golden_ref_v1 = hbsattn_reference_v1_base(q, k, v, cu_seqlens, cu_seqlens, block_mask, q_block_size, k_block_size, True, None, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)

    golden_ref_v2 = hbsattn_reference_v2_with_pytorch(q, k, v, cu_seqlens, cu_seqlens, block_mask, q_block_size, k_block_size, True, None, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)

    golden_ref_v3 = hbsattn_reference_v3_qkallfirst(q, k, v, cu_seqlens, cu_seqlens, block_mask, q_block_size, k_block_size, True, None, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)

    out = HBSAttention(q, k, v, cu_seqlens, max_seqlen, block_mask, q_block_size, k_block_size, True, None, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)
    
    
    print("golden_ref_v1", golden_ref_v1, torch.isnan(golden_ref_v1).any())
    print("golden_ref_v2", golden_ref_v2, torch.isnan(golden_ref_v2).any())
    print("golden_ref_v3", golden_ref_v3, torch.isnan(golden_ref_v3).any())
    
    # benchmarking start here
    benchmark({
        'golden': golden_ref_v2,
        'n_runs': 10,
        'n_warmup': 4,
        'name': 'hbsattn_reference_v1_base'
    }, hbsattn_reference_v1_base, q, k, v, cu_seqlens, cu_seqlens, block_mask, q_block_size, k_block_size, True, None, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)
    
    benchmark({
        'golden': golden_ref_v2,
        'n_runs': 10,
        'n_warmup': 4,
        'name': 'hbsattn_reference_v2_with_pytorch'
    }, hbsattn_reference_v2_with_pytorch, q, k, v, cu_seqlens, cu_seqlens, block_mask, q_block_size, k_block_size, True, None, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)
    
    benchmark({
        'golden': golden_ref_v2,
        'n_runs': 10,
        'n_warmup': 4,
        'name': 'hbsattn_reference_v3_qkallfirst'
    }, hbsattn_reference_v3_qkallfirst, q, k, v, cu_seqlens, cu_seqlens, block_mask, q_block_size, k_block_size, True, None, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)
    
    # benchmark({
    #     'golden': golden_ref_v1,
    #     'n_runs': 10,
    #     'n_warmup': 4,
    #     'name': 'HBSAttention'
    # }, HBSAttention, q, k, v, cu_seqlens, max_seqlen, block_mask, q_block_size, k_block_size, True, None, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)
    

    
    