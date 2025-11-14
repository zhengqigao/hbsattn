import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from hbsattn import HBSAttention
from hbsattn.reference import (
    hbsattn_reference_v1_base,
    hbsattn_reference_v2_with_pytorch,
    hbsattn_reference_v3_qkallfirst,
    hbsattn_reference_v4_hanlab_bsattn,
    hbsattn_reference_v5_flexattn,
)

from flash_attn import flash_attn_varlen_func

from hbsattn.utils import calculate_blocks, caculate_groups
from hbsattn.schedule import *
from hbsattn.benchmark import benchmark
import argparse
import json



# this file is used to test the speedup of our HBSAttention implementation compared to the reference implementation.
# We mainly compared with three reference implementations: 
#   (i) basic pytorch implementaiton
#   (iii) hanlab_block_sparse_attn: it requires block_size = 128
#   (iv) flex_attention: it requires (B,H,S,D) format input. 
# Thus, to make the test cases can be run for every method, we limit each sample has same seqlen, and block_size = 128. 

# We should note that our method is flexible to accept various input format (GQA, seq_len_q != seq_len_k, cu_q_seqlens != cu_k_seqlens, etc.).
# Please check `test_accuracy.py` for more details.


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--causal', action='store_true', default=False)
    parser.add_argument('--softmax_scale', type=float, default=None)
    parser.add_argument('--nruns', type=int, default=2)
    parser.add_argument('--nwarmup', type=int, default=1)
    parser.add_argument('--headdim', type=int, default=128)
    parser.add_argument('--unit_block', type=int, default=4, help='the number of blocks per unit, must be divisible by 4')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--sparse_ratio', type=float, default=0.3)
    parser.add_argument('--num_block_per_group', type=int, default=1, help='the number of blocks per group, used only for hbsattn (scheduling mode)')
    parser.add_argument('--block_size', type=int, default=16)
    parser.add_argument('--provide_info', action='store_true', default=False)
    args = parser.parse_args()
        
    nruns = args.nruns
    nwarmup = args.nwarmup
    causal = args.causal
    softmax_scale = args.softmax_scale
    headdim = args.headdim
    unit_block = args.unit_block
    nhead_k = args.nheads
    nhead_q = args.nheads
    num_block_per_group = args.num_block_per_group
    batch_size = args.batch_size
    q_block_size = args.block_size
    k_block_size = args.block_size
    
    device = torch.cuda.current_device()
    dtype = torch.bfloat16

    cu_k_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * unit_block * q_block_size
    max_k_seqlen = int((cu_k_seqlens[1:] - cu_k_seqlens[:-1]).max().item())
    k_seqlen = cu_k_seqlens[-1].item()
    
    cu_q_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * unit_block * k_block_size
    max_q_seqlen = int((cu_q_seqlens[1:] - cu_q_seqlens[:-1]).max().item())
    q_seqlen = cu_q_seqlens[-1].item()
    
    q = torch.randn(q_seqlen, nhead_q, headdim, device=device, dtype=dtype)
    k = torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)
    v =  torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)

    # the following information is needed for our HBSAttention implementation. You don't need to change that, providing cu_q/k_seqlens and q/k_blocksize is enough.
    num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
    num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)


    # Important: note if block_mask is not set properly, then it is possible for a q block not to attend to any k block, and cause the output to be NaN. When this happen, our implementation will return 0 instead of NaN (this is also what flashattn does when seq_len_q != seqlen_k).
    # block_mask = (torch.rand(nhead_k, num_q_block, num_k_block, device=device) < 1 - args.sparse_ratio).to(torch.bool)
    # for i in range(num_q_block):
    #     batch_idx = q_block_to_batch[i]
    #     first_k_block_idx_in_the_same_batch = None
    #     for j in range(len(k_block_to_batch)):
    #         if k_block_to_batch[j] == batch_idx:
    #             first_k_block_idx_in_the_same_batch = j
    #             break
    #     block_mask[:,i,first_k_block_idx_in_the_same_batch] = True # this can make sure q will attend to the first k block in the same batch.
    # assert torch.sum(block_mask, dim=-1).all() == True, "at least one k block is needed for each q."

    # print(f"block_mask: {block_mask}")
    
    block_mask = torch.zeros(nhead_k, num_q_block, num_k_block, device=device, dtype=torch.bool)
    for i in range(num_q_block):
        remainder = i % 4
        wrk = i // 4
        if remainder == 0:
            block_mask[:,i,wrk * 4] = True
        elif remainder == 1:
            block_mask[:,i,wrk * 4 + 0] = True
            block_mask[:,i,wrk * 4 + 2] = True
        if remainder == 2:
            block_mask[:,i,wrk * 4 + 1] = True
        elif remainder == 3:
            block_mask[:,i,wrk * 4 + 1] = True
            block_mask[:,i,wrk * 4 + 3] = True
    
    print(f"block_mask: {block_mask}")
    
    golden_res = hbsattn_reference_v1_base(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    #  The `calculate_blocks` function returns num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block
    #  These information is one-time calcualtion for the same sequence to pass through all attention layers, so can be amortized.
    #  Note when the input change, these information should be recalculated.
    #  In other words, they cannot be amortized across different input sequences, but can across different attention layers.
    #  Providing the args.provide_info can give us two different ways of measuring runtime: args.provide_info = True is closer to running it in a real LLM. 
    if not args.provide_info:
        our_auto_result = benchmark({
                    'golden': golden_res,
                    'n_runs': nruns,
                    'n_warmup': nwarmup,
                    'name': 'HBSAttention_auto_tilesize'
            }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'auto', num_block_per_group, )

        our_fix_result = benchmark({
                'golden': golden_res,
                'n_runs': nruns,
                'n_warmup': nwarmup,
                'name': 'HBSAttention_fix_tilesize'
            }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'fix', num_block_per_group, )

        our_scheduling_result = benchmark({
                'golden': golden_res,
                'n_runs': nruns,
                'n_warmup': nwarmup,
                'name': 'HBSAttention_scheduling'
            }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, base_schedule, num_block_per_group, )
    else:
        our_auto_result = benchmark({
                    'golden': golden_res,
                    'n_runs': nruns,
                    'n_warmup': nwarmup,
                    'name': 'HBSAttention_auto_tilesize'
            }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'auto', 
                                    num_block_per_group, 
                                    num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)


        our_fix_result = benchmark({
                'golden': golden_res,
                'n_runs': nruns,
                'n_warmup': nwarmup,
                'name': 'HBSAttention_fix_tilesize'
            }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'fix', 
                                    num_block_per_group, 
                                    num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

        # similarly, this informaion cannot be amortized across different input sequences, but can across different attention layers.
        num_q_group, cu_num_q_group, q_group_to_batch = caculate_groups(cu_num_q_block, num_block_per_group)

        our_scheduling_result = benchmark({
                'golden': golden_res,
                'n_runs': nruns,
                'n_warmup': nwarmup,
                'name': 'HBSAttention_scheduling'
            }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, base_schedule_optimized_v3, 
                                    num_block_per_group, 
                                    num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block,
                                    num_q_group, cu_num_q_group, q_group_to_batch)
