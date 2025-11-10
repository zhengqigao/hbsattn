import torch 
import einops
import math
import torch.nn.functional as F
from hbsattn.utils import calculate_blocks
from torch.nn.functional import scaled_dot_product_attention
import warnings

try:
    from block_sparse_attn import block_sparse_attn_func
except Exception as e:
    print(f"Importing block_sparse_attn failed with error: {e}")
    hanlab_block_sparse_attn_func = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except Exception as e:
    print(f"Importing FlexAttention failed with error: {e}")
    flex_attention = None
    create_block_mask = None


def hbsattn_reference_v1_base(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block =None , cu_q_block = None, q_block_to_batch = None, cu_num_q_block = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None, cu_num_k_block = None):
    
    if num_q_block is None or cu_q_block is None or q_block_to_batch is None or cu_num_q_block is None:
        num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
    if num_k_block is None or cu_k_block is None or k_block_to_batch is None or cu_num_k_block is None:
        num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)
        
    assert len(cu_q_seqlens) == len(cu_k_seqlens) and cu_q_seqlens.ndim == 1 and cu_k_seqlens.ndim == 1, "cu_q_seqlens and cu_k_seqlens must be 1D tensors of same length, indicating the start and end indices of each q/k sample. Their length equals the total number of samples + 1."
    
    batch_size = len(cu_q_seqlens) - 1 # number of samples
    
    nhead_q = q.shape[1]
    nhead_k = k.shape[1]
    assert nhead_q % nhead_k == 0, "nhead_q must be divisible by nhead_k (for GQA)"
    shared_ratio = nhead_q // nhead_k
    
    headdim = q.shape[2]
    seq_len_q = q.shape[0]
    seq_len_k = k.shape[0]
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5

    output = torch.empty_like(q)
    
    for block_idx in range(num_q_block):
        start_q = cu_q_block[block_idx]
        end_q = cu_q_block[block_idx + 1]

        q_block = q[start_q:end_q] # shape (block_seq_len, nhead, headdim)
        
        qk = torch.einsum('mhd,shd->msh', q_block.float(), k.float().repeat_interleave(shared_ratio, dim=1)) # shape (block_seq_len, seq_len_k, nhead_q)
        qk *= softmax_scale
        
        # First: in the same batch mask (same in both the block_seq_len and nhead_q dimension). 
        in_batch_mask = torch.zeros((seq_len_k), dtype=torch.bool, device=q.device)
        batch_idx = q_block_to_batch[block_idx]
        batch_k_start_idx = cu_k_seqlens[batch_idx]
        batch_k_end_idx = cu_k_seqlens[batch_idx + 1]
        in_batch_mask[batch_k_start_idx:batch_k_end_idx] = True
        
        # Second: causal mask, True means the position is allowed to be attended to. (same in the nhead_q dimension)
        current_causal_mask = torch.ones((qk.shape[0], qk.shape[1]), dtype=torch.bool, device=block_mask.device)
        if causal:
            batch_q_start_idx = cu_q_seqlens[batch_idx]
            batch_q_end_idx = cu_q_seqlens[batch_idx + 1]
            
            current_q_seq_len = batch_q_end_idx - batch_q_start_idx
            current_k_seq_len = batch_k_end_idx - batch_k_start_idx
            offset = current_k_seq_len - current_q_seq_len

            for i in range(qk.shape[0]):
                for j in range(qk.shape[1]):
                    q_index_in_batch = cu_q_block[block_idx] + i - batch_q_start_idx
                    k_index_in_batch = j - batch_k_start_idx
                    if q_index_in_batch + offset < k_index_in_batch:
                        current_causal_mask[i, j] = False
        
        # Finally: block mask, True means the block is needed to be attended to. (same in the block_seq_len dimension)
        current_block_mask = torch.empty((seq_len_k, nhead_q), dtype=torch.bool, device=block_mask.device)
        for i in range(nhead_q):
            head_k_idx = i // shared_ratio
            for j in range(num_k_block):
                bm = block_mask[head_k_idx,block_idx,j]
                start_idx = cu_k_block[j]
                end_idx = cu_k_block[j+1]
                current_block_mask[start_idx:end_idx, i] = bm
        
        total_mask = in_batch_mask.view(1,-1,1) & current_causal_mask.unsqueeze(-1) & current_block_mask.unsqueeze(0)
        
        
        qk = qk.masked_fill(total_mask.logical_not(), float('-inf'))


        p = F.softmax(qk, dim=1)
        
        # if a q attend to no k, then the corresponding values in p will be NaN. Set them to 0.0. This easily can happen when seqlen_q > seqlen_k: see FlashAttention Github `2.1: Change behavior of causal flag`.
        p = torch.nan_to_num(p, nan=0.0) 
        
        out = torch.einsum('msh,shd->mhd', p, v.float().repeat_interleave(shared_ratio, dim=1))
        output[start_q:end_q] = out 
    
    # if torch.isnan(output).any():
    #     warnings.warn("Warning: NaN detected in output of hbsattn_reference_v1_base. It is possible if the block mask makes a q block not attend to any k block.")
    #     nan_indices = torch.isnan(output).nonzero(as_tuple=True)
    #     print(f"NaN found at indices: {nan_indices}")
    return output

def hbsattn_reference_v2_with_pytorch(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block =None , cu_q_block = None, q_block_to_batch = None, cu_num_q_block = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None, cu_num_k_block = None):
    
    if num_q_block is None or cu_q_block is None or q_block_to_batch is None or cu_num_q_block is None:
        num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
    if num_k_block is None or cu_k_block is None or k_block_to_batch is None or cu_num_k_block is None:
        num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)
    
    assert len(cu_q_seqlens) == len(cu_k_seqlens) and cu_q_seqlens.ndim == 1 and cu_k_seqlens.ndim == 1, "cu_q_seqlens and cu_k_seqlens must be 1D tensors of same length, indicating the start and end indices of each q/k sample. Their length equals the total number of samples + 1."
    
    batch_size = len(cu_q_seqlens) - 1 # number of samples
    
    nhead_q = q.shape[1]
    nhead_k = k.shape[1]
    assert nhead_q % nhead_k == 0, "nhead_q must be divisible by nhead_k (for GQA)"
    shared_ratio = nhead_q // nhead_k
    
    headdim = q.shape[2]
    seq_len_q = q.shape[0]
    seq_len_k = k.shape[0]
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5

    output = torch.empty_like(q)
    
    for block_idx in range(num_q_block):
        for head_q_idx in range(nhead_q):
            head_k_idx = head_q_idx // shared_ratio
            
            q_block = q[cu_q_block[block_idx]:cu_q_block[block_idx + 1], head_q_idx, :] # shape (block_seq_len, headdim)
            
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
            if causal:
                batch_q_start_idx = cu_q_seqlens[batch_idx]
                batch_q_end_idx = cu_q_seqlens[batch_idx + 1]
                
                current_q_seq_len = batch_q_end_idx - batch_q_start_idx
                current_k_seq_len = batch_k_end_idx - batch_k_start_idx
                offset = current_k_seq_len - current_q_seq_len # in most cases, cu_q_seqlens == cu_k_seqlens, so offset is always zero.

                for i in range(q_block.shape[0]):
                    for j in range(seq_len_k):
                        q_index_in_batch = cu_q_block[block_idx] + i - batch_q_start_idx
                        k_index_in_batch = j - batch_k_start_idx
                        if q_index_in_batch + offset < k_index_in_batch:
                            current_mask[i, j] = False
   
            # Finally, add block mask, the k blocks that are not needed for the current q block should be ignored.
            for j in range(num_k_block):
                if not block_mask[head_k_idx,block_idx,j]:
                    start_idx = cu_k_block[j]
                    end_idx = cu_k_block[j+1]
                    current_mask[:, start_idx:end_idx] = False
            
            out = scaled_dot_product_attention(
               query = q_block.unsqueeze(0), # (1, block_seq_len, headdim),
               key = k[:,head_k_idx, :].unsqueeze(0), # (1, seq_len_k, headdim),
               value = v[:,head_k_idx, :].unsqueeze(0), # (1, seq_len_k, headdim),
               attn_mask = current_mask, # shape (block_seq_len, seq_len_k)
               is_causal = False, # we have incoporated all constraints in the current_mask.
               scale = softmax_scale,
               )
            output[cu_q_block[block_idx]:cu_q_block[block_idx + 1], head_q_idx, :] = out
    
    if torch.isnan(output).any():
        warnings.warn("Warning: NaN detected in output of hbsattn_reference_v2_with_pytorch. It is possible if the block mask makes a q block not attend to any k block.")
        nan_indices = torch.isnan(out).nonzero(as_tuple=True)
        print(f"NaN found at indices: {nan_indices}")
    return output

def hbsattn_reference_v3_qkallfirst(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block =None , cu_q_block = None, q_block_to_batch = None, cu_num_q_block = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None, cu_num_k_block = None):
    
    if num_q_block is None or cu_q_block is None or q_block_to_batch is None or cu_num_q_block is None:
        num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
    if num_k_block is None or cu_k_block is None or k_block_to_batch is None or cu_num_k_block is None:
        num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)
    
    assert len(cu_q_seqlens) == len(cu_k_seqlens) and cu_q_seqlens.ndim == 1 and cu_k_seqlens.ndim == 1, "cu_q_seqlens and cu_k_seqlens must be 1D tensors of same length, indicating the start and end indices of each q/k sample. Their length equals the total number of samples + 1."
    
    batch_size = len(cu_q_seqlens) - 1 # number of samples
    
    nhead_q = q.shape[1]
    nhead_k = k.shape[1]
    assert nhead_q % nhead_k == 0, "nhead_q must be divisible by nhead_k (for GQA)"
    shared_ratio = nhead_q // nhead_k
    
    headdim = q.shape[2]
    seq_len_q = q.shape[0]
    seq_len_k = k.shape[0]
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5

    qk = torch.einsum('nhd,shd->nsh', q.float(), k.float().repeat_interleave(shared_ratio, dim=1)) * softmax_scale # shape (seq_len_q, seq_len_k, nhead_q)
    
    # construct a large overall mask named total_mask
    # First, construct the in_batch_mask and causl_mask together at the same time (same in the nhead_q dimension)
    in_batch_and_causal_mask = torch.zeros((seq_len_q, seq_len_k,), dtype=torch.bool, device=q.device)
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
                if not causal:
                    in_batch_and_causal_mask[i, j] = True
                elif i - batch_q_start_idx + offset >= j - batch_k_start_idx: 
                    in_batch_and_causal_mask[i, j] = True
    
    # Second, construct the block_mask
    expanded_block_mask = torch.zeros((seq_len_q, seq_len_k, nhead_q), dtype=torch.bool, device=q.device)
    for i in range(nhead_q):
        head_k_idx = i // shared_ratio
        for j in range(num_q_block):
            for k in range(num_k_block):
                if block_mask[head_k_idx,j,k]:
                    start_q_index = cu_q_block[j]
                    end_q_index = cu_q_block[j+1]
                    start_k_index = cu_k_block[k]
                    end_k_index = cu_k_block[k+1]
                    expanded_block_mask[start_q_index:end_q_index, start_k_index:end_k_index, i] = True
    
    total_mask = in_batch_and_causal_mask.unsqueeze(-1) & expanded_block_mask
    
    # Check if any elements in total_mask.sum(dim=1) are equal to zero
    # index = (total_mask.sum(dim=1) == 0) # total_mask[index[0],:,index[1]] contains all False.
    # print(f"total_mask.shape: {total_mask.shape}")
    # print(f"index.shape: {index.shape}")
    
    # now masking and calculate output
    qk = qk.masked_fill(total_mask.logical_not(), float('-inf'))
    p = F.softmax(qk, dim=1) # shape (seq_len_q, seq_len_k, nhead_q)
    
    # if a q attend to no k, then the corresponding values in p will be NaN. Set them to 0.0. This easily can happen when seqlen_q > seqlen_k: see FlashAttention Github `2.1: Change behavior of causal flag`.
    p = torch.nan_to_num(p, nan=0.0) 
    
    out = torch.einsum('nsh,shd->nhd', p, v.float().repeat_interleave(shared_ratio, dim=1)).to(q.dtype)
    
    if torch.isnan(out).any():
        warnings.warn("Warning: NaN detected in output of hbsattn_reference_v3_qkallfirst. It is possible if the block mask makes a q block not attend to any k block.")
        nan_indices = torch.isnan(out).nonzero(as_tuple=True)
        print(f"NaN found at indices: {nan_indices}")
    
    return out


def hbsattn_reference_v4_hanlab_bsattn(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, causal, softmax_scale,):
    
    nhead_q = q.shape[1]
    max_seqlen_q = torch.max(cu_q_seqlens[1:] - cu_q_seqlens[:-1]).item()
    max_seqlen_k = torch.max(cu_k_seqlens[1:] - cu_k_seqlens[:-1]).item()    

    out = block_sparse_attn_func(
    q, 
    k,
    v,
    cu_q_seqlens, 
    cu_k_seqlens,
    head_mask_type = torch.ones(nhead_q, dtype=torch.int32, device=q.device),
    streaming_info = None,
    base_blockmask = block_mask,
    max_seqlen_q_ = max_seqlen_q, 
    max_seqlen_k_ = max_seqlen_k,
    p_dropout = 0.0,
    deterministic=False,
    softmax_scale=softmax_scale,
    is_causal=causal,
    exact_streaming=False,
    return_attn_probs=False,
    )
    return out 


def hbsattn_reference_v5_flexattn(q_padded, k_padded, v_padded, block_mask, block_size, causal, scale):
    
    def mask_mod_causal(b, h, q_idx, kv_idx):

        return (q_idx >= kv_idx) and block_mask[b, h, q_idx // block_size, kv_idx // block_size]
    
    def mask_mod_nocausal(b, h, q_idx, kv_idx):
        return block_mask[b, h, q_idx // block_size, kv_idx // block_size]
    
    
    B = q_padded.shape[0]
    H = q_padded.shape[1]
    S = q_padded.shape[2]
    D = q_padded.shape[3]
    
    if causal:
        flex_block_mask = create_block_mask(
            mask_mod_causal,
            B=B, 
            H=H,    # nhead
            Q_LEN=S, 
            KV_LEN=S,
            BLOCK_SIZE=block_size
        )
        
        # Use flex_attention with the optimized block mask
        output = flex_attention(q_padded, k_padded, v_padded, block_mask=flex_block_mask, scale=scale)
    else:
        flex_block_mask = create_block_mask(
            mask_mod_nocausal,
            B=B, 
            H=H,    # nhead
            Q_LEN=S, 
            KV_LEN=S,
            BLOCK_SIZE=block_size
        )
        output = flex_attention(q_padded, k_padded, v_padded, block_mask=flex_block_mask, scale=scale)
    # # reshape to [Seqlen, nhead, headdim] for consistency
    # output = output.reshape(-1, H, D)
    return output 