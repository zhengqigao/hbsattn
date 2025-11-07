import torch
from hbsattn.fwd_triton import HBSAttention
from hbsattn.reference import hbsattn_reference_v1_base, hbsattn_reference_v2_with_pytorch, hbsattn_reference_v3_qkallfirst
from hbsattn.utils import calculate_blocks
from hbsattn.benchmark import benchmark



if __name__ == "__main__":

    nruns = 2
    nwarmup = 1
    
    device = torch.cuda.current_device()

    causal = False
    softmax_scale = None
    
    dtype = torch.float32
    
    cu_k_seqlens = torch.tensor([0,16,], dtype=torch.int32, device=device) # [0, 32, 64, 96, 128, 160] # , 61, 100, 134, 157
    max_k_seqlen = int((cu_k_seqlens[1:] - cu_k_seqlens[:-1]).max().item())
    k_seqlen = cu_k_seqlens[-1].item()
    
    cu_q_seqlens = torch.tensor([0,16,], dtype=torch.int32, device=device) # [0, 32, 64, 96, 128, 160]
    max_q_seqlen = int((cu_q_seqlens[1:] - cu_q_seqlens[:-1]).max().item())
    q_seqlen = cu_q_seqlens[-1].item()
    
    nhead_k = 1
    nhead_q = 1
    
    assert nhead_q % nhead_k == 0, "nhead_q must be divisible by nhead_k (for GQA)"
    
    headdim = 18
    q_block_size = 16
    k_block_size = 16

    
    q = torch.ones(q_seqlen, nhead_q, headdim, device=device, dtype=dtype)
    k = torch.ones(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)
    v =  torch.ones(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)
    # v = torch.arange(k_seqlen,device=device, dtype=dtype).view(k_seqlen, 1, 1).repeat(1, nhead_k, headdim)
    # Set print precision for PyTorch tensors to display 7 decimal places
    # torch.set_printoptions(precision=7, sci_mode=False)
    # print("v[:,0,0]", v[:,0,0])
    
    # # INSERT_YOUR_CODE
    # # Load q, k, v, and block_mask that were saved as "q.pt", "k.pt", "v.pt", "block_mask.pt"
    # # Make sure they're loaded to the correct device and dtype
    # # 修复TypeError: 'int' object is not callable: map_location 参数应该传字符串而不是 device id/int
    # # 推荐直接用 map_location=f"cuda:{device}" 如果 device 是 int，否则可直接用 torch.device
    # device_str = f"cuda:{device}" if isinstance(device, int) else device
    # q = torch.load("q.pt", map_location=device_str).to(dtype)
    # k = torch.load("k.pt", map_location=device_str).to(dtype)
    # v = torch.load("v.pt", map_location=device_str).to(dtype)
    # block_mask = torch.load("block_mask.pt", map_location=device_str)
    
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
    print("block_mask", block_mask)
    assert torch.sum(block_mask, dim=-1).all() == True, "at least one k block is needed for each q."
    
    # run once to get a golden reference
    golden_ref_v1 = hbsattn_reference_v1_base(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    golden_ref_v2 = hbsattn_reference_v2_with_pytorch(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    golden_ref_v3 = hbsattn_reference_v3_qkallfirst(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    out = HBSAttention(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
    

    print("golden_ref_v1", golden_ref_v1, torch.isnan(golden_ref_v1).any())
    print("golden_ref_v2", golden_ref_v2, torch.isnan(golden_ref_v2).any())
    print("golden_ref_v3", golden_ref_v3, torch.isnan(golden_ref_v3).any())
    print("out", out, torch.isnan(out).any())
    # Find the index of the most different value between out and golden_ref_v1, and show their values
    diff = (out - golden_ref_v1).abs()
    max_diff = diff.max()
    max_idx = (diff == max_diff).nonzero(as_tuple=True)


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
        'name': 'hbsattn_reference_v3_qkallfirst'
    }, hbsattn_reference_v3_qkallfirst, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
    
    benchmark({
        'golden': golden_ref_v1,
        'n_runs': nruns,
        'n_warmup': nwarmup,
        'name': 'HBSAttention'
    }, HBSAttention, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
    

    
    