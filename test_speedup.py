import torch
from hbsattn import HBSAttention
from hbsattn.reference import hbsattn_reference_v1_base, hbsattn_reference_v2_with_pytorch, hbsattn_reference_v3_qkallfirst, hbsattn_reference_v4_hanlab_bsattn
from hbsattn.utils import calculate_blocks
from hbsattn.benchmark import benchmark
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--causal', action='store_true', default=False)
    parser.add_argument('--softmax_scale', type=float, default=None)
    parser.add_argument('--nruns', type=int, default=2)
    parser.add_argument('--nwarmup', type=int, default=1)
    parser.add_argument('--headdim', type=int, default=16)
    parser.add_argument('--unit_seqlen', type=int, default=256)
    args = parser.parse_args()
    
    nruns = args.nruns
    nwarmup = args.nwarmup
    causal = args.causal
    softmax_scale = args.softmax_scale
    headdim = args.headdim
    unit_seqlen = args.unit_seqlen
    
    device = torch.cuda.current_device()
    dtype = torch.bfloat16
    
    # cu_k_seqlens = torch.tensor([0,32, 61, 100, 134, 157, 201, 253, 260], dtype=torch.int32, device=device) # [0, 32, 64, 96, 128, 160] # , 61, 100, 134, 157

    batch_size = 8
    cu_k_seqlens = torch.arange(0,batch_size+1, dtype=torch.int32, device=device) * unit_seqlen
    max_k_seqlen = int((cu_k_seqlens[1:] - cu_k_seqlens[:-1]).max().item())
    k_seqlen = cu_k_seqlens[-1].item()
    
    # cu_q_seqlens = torch.tensor([0,32, 64, 96, 128, 160,165, 170, 280], dtype=torch.int32, device=device) # [0, 32, 64, 96, 128, 160]
    cu_q_seqlens = torch.arange(0,batch_size+1, dtype=torch.int32, device=device) * unit_seqlen
    max_q_seqlen = int((cu_q_seqlens[1:] - cu_q_seqlens[:-1]).max().item())
    q_seqlen = cu_q_seqlens[-1].item()
    
    nhead_k = 1
    nhead_q = 1
    
    assert nhead_q % nhead_k == 0, "nhead_q must be divisible by nhead_k (for GQA)"
    
    q_block_size = 128
    k_block_size = 128
    
    q = torch.ones(q_seqlen, nhead_q, headdim, device=device, dtype=dtype)
    k = torch.ones(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)
    v =  torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)

    
    num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
    num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)


    # Important: note if block_mask is not set properly, then it is possible for a q block not to attend to any k block, and cause the output to be NaN.
    block_mask = (torch.rand(nhead_k, num_q_block, num_k_block, device=device) < 0.7).to(torch.bool)
    for i in range(num_q_block):
        batch_idx = q_block_to_batch[i]
        first_k_block_idx_in_the_same_batch = None
        for j in range(len(k_block_to_batch)):
            if k_block_to_batch[j] == batch_idx:
                first_k_block_idx_in_the_same_batch = j
                break
        block_mask[:,i,first_k_block_idx_in_the_same_batch] = True # this can make sure q will attend to the first k block in the same batch.
    # block_mask = block_mask.fill_(1).contiguous()
    print("block_mask.shape", block_mask.shape)
    assert torch.sum(block_mask, dim=-1).all() == True, "at least one k block is needed for each q."
    
    # construct block mask for hanlab_block_sparse_attn
    block_mask_hanlab_bsattn = torch.empty(batch_size, nhead_k, unit_seqlen//q_block_size, unit_seqlen//k_block_size, device=device, dtype=torch.bool)
    print("block_mask_hanlab_bsattn.shape", block_mask_hanlab_bsattn.shape)
    for i in range(batch_size):
        for j in range(nhead_k):
            for t1 in range(unit_seqlen//q_block_size):
                for t2 in range(unit_seqlen//k_block_size):
                    q_block_idx = i * (unit_seqlen//q_block_size) + t1
                    k_block_idx = i * (unit_seqlen//k_block_size) + t2
                    print("i, j, t1, t2", i, j, t1, t2)
                    print("q_block_idx", q_block_idx)
                    print("k_block_idx", k_block_idx)
                    block_mask_hanlab_bsattn[i,j,t1,t2] = block_mask[j,q_block_idx,k_block_idx]

    
    # run once to get a golden reference
    golden_ref_v1 = hbsattn_reference_v1_base(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    # golden_ref_v2 = hbsattn_reference_v2_with_pytorch(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    # golden_ref_v4 = hbsattn_reference_v4_hanlab_bsattn(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask_hanlab_bsattn, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    # out_auto_tilesize = HBSAttention(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'auto', num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
    
    # out_fix_tilesize = HBSAttention(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'fix', num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)


    # benchmarking start here
    benchmark({
        'golden': golden_ref_v1,
        'n_runs': nruns,
        'n_warmup': nwarmup,
        'name': 'hbsattn_reference_v1_base'
    }, hbsattn_reference_v1_base, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
    
    benchmark({
        'golden': golden_ref_v1,
        'n_runs': nruns,
        'n_warmup': nwarmup,
        'name': 'hbsattn_reference_v2_with_pytorch'
    }, hbsattn_reference_v2_with_pytorch, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
    

    benchmark({
            'golden': golden_ref_v1,
            'n_runs': nruns,
            'n_warmup': nwarmup,
            'name': 'HBSAttention_auto_tilesize'
    }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'auto', num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)


    benchmark({
        'golden': golden_ref_v1,
        'n_runs': nruns,
        'n_warmup': nwarmup,
        'name': 'HBSAttention_fix_tilesize'
    }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'fix', num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
        
    
    benchmark({
                'golden': golden_ref_v1,
                'n_runs': nruns,
                'n_warmup': nwarmup,
                'name': 'HBSAttention_hanlab_bsattn'
    }, hbsattn_reference_v4_hanlab_bsattn, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask_hanlab_bsattn, causal, softmax_scale)


    
    