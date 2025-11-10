import torch
from hbsattn import HBSAttention
from hbsattn.reference import (
    hbsattn_reference_v1_base,
    hbsattn_reference_v2_with_pytorch,
    hbsattn_reference_v3_qkallfirst,
    hbsattn_reference_v4_hanlab_bsattn,
    hbsattn_reference_v5_flexattn,
)

from hbsattn.utils import calculate_blocks
from hbsattn.benchmark import benchmark
import argparse
import json
import os 


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
    parser.add_argument('--unit_seqlen', type=int, default=256)
    parser.add_argument('--nheads', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_benchmark_to_file', type=str, default = 'benchmark_results.json')
    args = parser.parse_args()
    
    nruns = args.nruns
    nwarmup = args.nwarmup
    causal = args.causal
    softmax_scale = args.softmax_scale
    headdim = args.headdim
    unit_seqlen = args.unit_seqlen
    nhead_k = args.nheads
    nhead_q = args.nheads
    batch_size = args.batch_size
    
    q_block_size = 128 # we fix to block size 128, since block_sparse_attn from Han lab only support block size 128 for comparing speedup.
    k_block_size = 128
    assert q_block_size == k_block_size == 128, "q_block_size and k_block_size must be the same."
    
    device = torch.cuda.current_device()
    dtype = torch.bfloat16

    cu_k_seqlens = torch.arange(0,batch_size+1, dtype=torch.int32, device=device) * unit_seqlen
    max_k_seqlen = int((cu_k_seqlens[1:] - cu_k_seqlens[:-1]).max().item())
    k_seqlen = cu_k_seqlens[-1].item()
    
    cu_q_seqlens = torch.arange(0,batch_size+1, dtype=torch.int32, device=device) * unit_seqlen
    max_q_seqlen = int((cu_q_seqlens[1:] - cu_q_seqlens[:-1]).max().item())
    q_seqlen = cu_q_seqlens[-1].item()
    
    q = torch.randn(q_seqlen, nhead_q, headdim, device=device, dtype=dtype)
    k = torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)
    v =  torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)

    # Construct the bached q,k,v for flex_attention.
    q_padded = q.reshape(batch_size, unit_seqlen, nhead_q, headdim).permute(0,2,1,3)
    k_padded = k.reshape(batch_size, unit_seqlen, nhead_k, headdim).permute(0,2,1,3)
    v_padded = v.reshape(batch_size, unit_seqlen, nhead_k, headdim).permute(0,2,1,3)
    
    
    tmpq_padded = torch.empty_like(q_padded)
    
    for i in range(batch_size):
        start_index = i * (unit_seqlen//q_block_size)
        end_index = start_index + (unit_seqlen//q_block_size)
        tmpq_padded[i] = q[start_index:end_index].permuate(0,1)
    assert torch.allclose(q_padded, tmpq_padded), "q_padded and tmpq_padded are not the same."
    
    

    # the following information is needed for our HBSAttention implementation. You don't need to change that, providing cu_q/k_seqlens and q/k_blocksize is enough.
    num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
    num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)


    # Important: note if block_mask is not set properly, then it is possible for a q block not to attend to any k block, and cause the output to be NaN. When this happen, our implementation will return 0 instead of NaN (this is also what flashattn does when seq_len_q != seqlen_k).
    block_mask = (torch.rand(nhead_k, num_q_block, num_k_block, device=device) < 0.7).to(torch.bool)
    for i in range(num_q_block):
        batch_idx = q_block_to_batch[i]
        first_k_block_idx_in_the_same_batch = None
        for j in range(len(k_block_to_batch)):
            if k_block_to_batch[j] == batch_idx:
                first_k_block_idx_in_the_same_batch = j
                break
        block_mask[:,i,first_k_block_idx_in_the_same_batch] = True # this can make sure q will attend to the first k block in the same batch.
    assert torch.sum(block_mask, dim=-1).all() == True, "at least one k block is needed for each q."
    
    # Convert the block_mask to the format for hanlab_block_sparse_attn.
    block_mask_hanlab_bsattn = torch.empty(batch_size, nhead_k, unit_seqlen//q_block_size, unit_seqlen//k_block_size, device=device, dtype=torch.bool)
    for i in range(batch_size):
        for j in range(nhead_k):
            for t1 in range(unit_seqlen//q_block_size):
                for t2 in range(unit_seqlen//k_block_size):
                    q_block_idx = i * (unit_seqlen//q_block_size) + t1
                    k_block_idx = i * (unit_seqlen//k_block_size) + t2
                    block_mask_hanlab_bsattn[i,j,t1,t2] = block_mask[j,q_block_idx,k_block_idx]

    
    # run once to get a golden reference
    golden_ref_v1 = hbsattn_reference_v1_base(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    # benchmarking all methods start here
    v1_result = benchmark({
        'golden': golden_ref_v1,
        'n_runs': nruns,
        'n_warmup': nwarmup,
        'name': 'hbsattn_reference_v1_base'
    }, hbsattn_reference_v1_base, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
    
    v2_result = benchmark({
        'golden': golden_ref_v1,
        'n_runs': nruns,
        'n_warmup': nwarmup,
        'name': 'hbsattn_reference_v2_with_pytorch'
    }, hbsattn_reference_v2_with_pytorch, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    v3_result = benchmark({
            'golden': golden_ref_v1,
            'n_runs': nruns,
            'n_warmup': nwarmup,
            'name': 'HBSAttention_auto_tilesize'
    }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'auto', num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    our_result = benchmark({
        'golden': golden_ref_v1,
        'n_runs': nruns,
        'n_warmup': nwarmup,
        'name': 'HBSAttention_fix_tilesize'
    }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, 'fix', num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    v4_result = benchmark({
                'golden': golden_ref_v1,
                'n_runs': nruns,
                'n_warmup': nwarmup,
                'name': 'HBSAttention_hanlab_bsattn'
    }, hbsattn_reference_v4_hanlab_bsattn, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask_hanlab_bsattn, causal, softmax_scale)


    v5_result = benchmark({
                'golden': golden_ref_v1,
                'n_runs': nruns,
                'n_warmup': nwarmup,
                'name': 'HBSAttention_flexattn'
    }, hbsattn_reference_v5_flexattn, q_padded, k_padded, v_padded, block_mask_hanlab_bsattn, q_block_size, causal, softmax_scale)
    

        
    
    # if save_benchmark_to_file is not empty, save the benchmark results to a file.
    print(f"ss,", args.save_benchmark_to_file)
    if args.save_benchmark_to_file:
        # Save all benchmark results in a dict for one-shot dump
        all_results = {
            "our_result": our_result,
            "v1_result": v1_result,
            "v2_result": v2_result,
            "v3_result": v3_result,
            "v4_result": v4_result,
            "v5_result": v5_result,
        }
        # Append all_results as a line-delimited JSON object to the file
        with open(args.save_benchmark_to_file, 'a') as f:
            f.write(json.dumps(all_results, indent=4))
            f.write('\n')
        print(f"Benchmark results appended to {args.save_benchmark_to_file}")