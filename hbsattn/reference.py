import torch 
import einops
import math
import torch.nn.functional as F
from hbsattn.utils import calculate_blocks
from torch.nn.functional import scaled_dot_product_attention
import warnings

def hbsattn_reference_v1_base(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block =None , cu_q_block = None, q_block_to_batch = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None):
    
    if num_q_block is None or cu_q_block is None or q_block_to_batch is None:
        num_q_block, cu_q_block, q_block_to_batch = calculate_blocks(cu_q_seqlens, q_block_size)
    if num_k_block is None or cu_k_block is None or k_block_to_batch is None:
        num_k_block, cu_k_block, k_block_to_batch = calculate_blocks(cu_k_seqlens, k_block_size)
        
    assert len(cu_q_seqlens) == len(cu_k_seqlens) and cu_q_seqlens.ndim == 1 and cu_k_seqlens.ndim == 1, "cu_q_seqlens and cu_k_seqlens must be 1D tensors of same length, indicating the start and end indices of each q/k sample. Their length equals the total number of samples + 1."
    
    batch_size = len(cu_q_seqlens) - 1 # number of samples
    nhead = q.shape[1]
    headdim = q.shape[2]
    seq_len_q = q.shape[0]
    seq_len_k = k.shape[0]
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5
    output = torch.empty_like(q)
    
    for block_idx in range(num_q_block):
        start_q = cu_q_block[block_idx]
        end_q = cu_q_block[block_idx + 1]

        q_block = q[start_q:end_q] # shape (block_seq_len, nhead, headdim)
        
        qk = torch.einsum('mhd,shd->msh', q_block, k) # shape (block_seq_len, seq_len_k, nhead)
        qk *= softmax_scale
        
        # First: in the same batch mask (same in both the block_seq_len and nhead dimension). 
        in_batch_mask = torch.zeros((seq_len_k), dtype=torch.bool, device=q.device)
        batch_idx = q_block_to_batch[block_idx]
        batch_k_start_idx = cu_k_seqlens[batch_idx]
        batch_k_end_idx = cu_k_seqlens[batch_idx + 1]
        in_batch_mask[batch_k_start_idx:batch_k_end_idx] = True
        
        # Second: causal mask, True means the position is allowed to be attended to. (same in the nhead dimension)
        current_causal_mask = torch.ones((qk.shape[0], qk.shape[1]), dtype=torch.bool, device=block_mask.device)
        batch_q_start_idx = cu_q_seqlens[batch_idx]
        batch_q_end_idx = cu_q_seqlens[batch_idx + 1]
        
        current_q_seq_len = batch_q_end_idx - batch_q_start_idx
        current_k_seq_len = batch_k_end_idx - batch_k_start_idx
        offset = current_k_seq_len - current_q_seq_len
        print(f"offset: {offset}")
        for i in range(qk.shape[0]):
            for j in range(qk.shape[1]):
                q_index_in_batch = cu_q_block[block_idx] + i - batch_q_start_idx
                k_index_in_batch = j - batch_k_start_idx
                if q_index_in_batch + offset < k_index_in_batch:
                    current_causal_mask[i, j] = False
        # print(f"in v1, block_idx={block_idx}, current_causal_mask & in_batch_mask.unsqueeze(0): {current_causal_mask & in_batch_mask.unsqueeze(0)}")
        # Finally: block mask, True means the block is needed to be attended to. (same in the block_seq_len dimension)
        current_block_mask = torch.empty((seq_len_k, nhead), dtype=torch.bool, device=block_mask.device)
        for i in range(nhead):
            for j in range(num_k_block):
                bm = block_mask[i,block_idx,j]
                start_idx = cu_k_block[j]
                end_idx = cu_k_block[j+1]
                current_block_mask[start_idx:end_idx, i] = bm
                print(f"i={i}, j={j}, start_idx: {start_idx}, end_idx: {end_idx}")
        
        total_mask = in_batch_mask.view(1,-1,1) & current_causal_mask.unsqueeze(-1) & current_block_mask.unsqueeze(0)
        
        
        qk = qk.masked_fill(total_mask.logical_not(), float('-inf'))
        
        # print(f"block_idx: {block_idx}, total_mask: {total_mask}, qk: {qk}")
        p = F.softmax(qk, dim=1)
        out = torch.einsum('msh,shd->mhd', p, v)
        output[start_q:end_q] = out 
    
    if torch.isnan(output).any():
        warnings.warn("Warning: NaN detected in output of hbsattn_reference_v1_base")
    return output

def hbsattn_reference_v2_with_pytorch(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block =None , cu_q_block = None, q_block_to_batch = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None):
    
    if num_q_block is None or cu_q_block is None or q_block_to_batch is None:
        num_q_block, cu_q_block, q_block_to_batch = calculate_blocks(cu_q_seqlens, q_block_size)
    if num_k_block is None or cu_k_block is None or k_block_to_batch is None:
        num_k_block, cu_k_block, k_block_to_batch = calculate_blocks(cu_k_seqlens, k_block_size)
    
    assert len(cu_q_seqlens) == len(cu_k_seqlens) and cu_q_seqlens.ndim == 1 and cu_k_seqlens.ndim == 1, "cu_q_seqlens and cu_k_seqlens must be 1D tensors of same length, indicating the start and end indices of each q/k sample. Their length equals the total number of samples + 1."
    
    batch_size = len(cu_q_seqlens) - 1
    nhead = q.shape[1]
    headdim = q.shape[2]
    seq_len_q = q.shape[0]
    seq_len_k = k.shape[0]
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5
    output = torch.empty_like(q)
    
    for block_idx in range(num_q_block):
        for head_idx in range(nhead):
            
            q_block = q[cu_q_block[block_idx]:cu_q_block[block_idx + 1], head_idx, :] # shape (block_seq_len, headdim)
            
            # construct the mask, initially set attend to every k.
            current_mask = torch.ones((q_block.shape[0], seq_len_k), dtype=torch.bool, device=q.device)
            
            # First, the k blocks that are not in the same batch as the `block_idx`-th q block should be ignored. 
            batch_idx = q_block_to_batch[block_idx]
            batch_k_start_idx = cu_k_seqlens[batch_idx]
            batch_k_end_idx = cu_k_seqlens[batch_idx + 1]
            current_mask[:,:batch_k_start_idx] = False
            current_mask[:,batch_k_end_idx:] = False
            
            # Second, add causality in the same batch, only <qi, kj> is allowed when i + offset <= j (offset=0 if seqlen_q == seqlen_k) 
            # See flash attention changelog og v2.1 for more details.
            batch_q_start_idx = cu_q_seqlens[batch_idx]
            batch_q_end_idx = cu_q_seqlens[batch_idx + 1]
            
            current_q_seq_len = batch_q_end_idx - batch_q_start_idx
            current_k_seq_len = batch_k_end_idx - batch_k_start_idx
            offset = current_k_seq_len - current_q_seq_len # in most cases, cu_q_seqlens == cu_k_seqlens, so offset is always zero.
            print(f"offset: {offset}")
            for i in range(q_block.shape[0]):
                for j in range(seq_len_k):
                    q_index_in_batch = cu_q_block[block_idx] + i - batch_q_start_idx
                    k_index_in_batch = j - batch_k_start_idx
                    if q_index_in_batch + offset < k_index_in_batch:
                        current_mask[i, j] = False
            # if head_idx == 0:
            #     print(f"in v2, block_idx={block_idx}, current_mask: {current_mask}")
                
            # Finally, add block mask, the k blocks that are not needed for the current q block should be ignored.
            for j in range(num_k_block):
                if not block_mask[head_idx,block_idx,j]:
                    start_idx = cu_k_block[j]
                    end_idx = cu_k_block[j+1]
                    current_mask[:, start_idx:end_idx] = False
            
            out = scaled_dot_product_attention(
               query = q_block.unsqueeze(0), # (1, block_seq_len, headdim),
               key = k[:,head_idx, :].unsqueeze(0), # (1, seq_len_k, headdim),
               value = v[:,head_idx, :].unsqueeze(0), # (1, seq_len_k, headdim),
               attn_mask = current_mask, # shape (block_seq_len, seq_len_k)
               is_causal = False, # we have incoporated all constraints in the current_mask.
               )
            output[cu_q_block[block_idx]:cu_q_block[block_idx + 1], head_idx, :] = out
    
    if torch.isnan(output).any():
        warnings.warn("Warning: NaN detected in output of hbsattn_reference_v2_with_pytorch")
    
    return output

def hbsattn_reference_v3_qkallfirst(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block =None , cu_q_block = None, q_block_to_batch = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None):
    
    if num_q_block is None or cu_q_block is None or q_block_to_batch is None:
        num_q_block, cu_q_block, q_block_to_batch = calculate_blocks(cu_q_seqlens, q_block_size)
    if num_k_block is None or cu_k_block is None or k_block_to_batch is None:
        num_k_block, cu_k_block, k_block_to_batch = calculate_blocks(cu_k_seqlens, k_block_size)
    
    assert len(cu_q_seqlens) == len(cu_k_seqlens) and cu_q_seqlens.ndim == 1 and cu_k_seqlens.ndim == 1, "cu_q_seqlens and cu_k_seqlens must be 1D tensors of same length, indicating the start and end indices of each q/k sample. Their length equals the total number of samples + 1."
    
    batch_size = len(cu_q_seqlens) - 1
    nhead = q.shape[1]
    headdim = q.shape[2]
    seq_len_q = q.shape[0]
    seq_len_k = k.shape[0]
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5

    qk = torch.einsum('nhd,shd->nsh', q, k) # shape (seq_len_q, seq_len_k, nhead)
    
    # construct a large overall mask named total_mask
    # First, construct the in_batch_mask and causl_mask together at the same time
    in_batch_and_causal_mask = torch.zeros((seq_len_q, seq_len_k, nhead), dtype=torch.bool, device=q.device)
    for sample_idx in range(batch_size):
        batch_k_start_idx = cu_k_seqlens[sample_idx]
        batch_k_end_idx = cu_k_seqlens[sample_idx + 1]
        batch_q_start_idx = cu_q_seqlens[sample_idx]
        batch_q_end_idx = cu_q_seqlens[sample_idx + 1]

        current_q_seq_len = batch_q_end_idx - batch_q_start_idx
        current_k_seq_len = batch_k_end_idx - batch_k_start_idx
        offset = current_k_seq_len - current_q_seq_len 
        
        for i in range(batch_q_start_idx, batch_q_end_idx):
            for j in range(batch_k_start_idx, batch_k_end_idx):
                if i + offset >= j: 
                    in_batch_and_causal_mask[i, j, :] = True
    
    # Second, construct the block_mask
    expanded_block_mask = torch.zeros((seq_len_q, seq_len_k, nhead), dtype=torch.bool, device=q.device)
    for i in range(nhead):
        for j in range(num_q_block):
            for k in range(num_k_block):
                if block_mask[i,j,k]:
                    start_q_index = cu_q_block[j]
                    end_q_index = cu_q_block[j+1]
                    start_k_index = cu_k_block[k]
                    end_k_index = cu_k_block[k+1]
                    expanded_block_mask[start_q_index:end_q_index, start_k_index:end_k_index, i] = True
    
    total_mask = in_batch_and_causal_mask & expanded_block_mask
    
    # Check if any elements in total_mask.sum(dim=1) are equal to zero
    # index = (total_mask.sum(dim=1) == 0) # total_mask[index[0],:,index[1]] contains all False.
    # print(f"total_mask.shape: {total_mask.shape}")
    # print(f"index.shape: {index.shape}")
    
    # now masking and calculate output
    qk = qk.masked_fill(total_mask.logical_not(), float('-inf'))
    p = F.softmax(qk, dim=1) # shape (seq_len_q, seq_len_k, nhead)
    
    # if index.numel():
    #     for i in range(index.shape[0]):
    #         for j in range(index.shape[1]):
    #             if index[i,j]:
    #                 p[i,:,j] = 0
    
    out = torch.einsum('nsh,shd->nhd', p, v)
    
    if torch.isnan(out).any():
        warnings.warn("Warning: NaN detected in output of hbsattn_reference_v3_qkallfirst")
    
    return out