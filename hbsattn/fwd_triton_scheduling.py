
import torch
import math 
import triton
import triton.language as tl
import torch.nn.functional as F
from hbsattn.utils import calculate_blocks

__all__ = ['_forward_scheduling']

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
        
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=4),
    ],
    key=['BLOCK_M', 'BLOCK_N'],
)
@triton.jit
def _fwd_kernel(
    q,
    k,
    v,
    cu_q_seqlens,
    cu_k_seqlens,
    causal,
    softmax_scale,
    cu_q_block, 
    cu_k_block,
    q_block_to_batch,
    cu_num_k_block,
    head_q_to_k_ratio,
    block_mask,
    q_assignment,
    q_group_to_batch,
    cu_q_group,
    out,
    lse,
    tmp, # See flash_attn_trion.py and flash_attn_triton_og.py 
    stride_q_s,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_b_nh,
    stride_b_nq,
    stride_b_nk,
    stride_assign_nh,
    stride_assign_ng,
    stride_assign_nb,
    stride_o_s,
    stride_o_h,
    stride_o_d,
    stride_lse_s,
    stride_lse_h,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    EVEN_SEQ_QBLOCK: tl.constexpr,
    EVEN_SEQ_KBLOCK: tl.constexpr,
):
    off_head_q = tl.program_id(1)
    off_head_k = off_head_q // head_q_to_k_ratio
    
    off_q_block = tl.program_id(0)
    off_dim = tl.arange(0, BLOCK_DIM)
    
    start_m = tl.load(cu_q_block + off_q_block)
    end_m = tl.load(cu_q_block + off_q_block + 1)
    off_m = start_m + tl.arange(0, BLOCK_M)
    
    # load the q block
    q_ptr = q + off_m[:, None] * stride_q_s + off_head_q * stride_q_h + off_dim[None, :] * stride_q_d
    if EVEN_SEQ_QBLOCK:
        if EVEN_HEADDIM:
            q_block = tl.load(q_ptr)
        else:
            q_block = tl.load(q_ptr, mask=off_dim[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q_block = tl.load(q_ptr, mask=off_m[:, None] < end_m, other=0.0)
        else:
            q_block = tl.load(q_ptr, mask=(off_m[:, None] < end_m) & (off_dim[None, :] < headdim), other=0.0)
    

    # accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=tl.float32)
    
    # tmp_ptr = tmp + off_head_q * stride_lse_h + off_m * stride_lse_s
    
    # batch index 
    batch_idx = tl.load(q_block_to_batch + off_q_block)
    
    # get offset and q and k start/end indices in seq
    batch_q_start_idx = tl.load(cu_q_seqlens + batch_idx)
    batch_q_end_idx = tl.load(cu_q_seqlens + batch_idx + 1)
    batch_k_start_idx = tl.load(cu_k_seqlens + batch_idx)
    batch_k_end_idx = tl.load(cu_k_seqlens + batch_idx + 1)
    offset = batch_k_end_idx - batch_k_start_idx - (batch_q_end_idx - batch_q_start_idx)

    # k block loop, start from the same batch as the q block, and end at the last k block in the same batch.
    k_block_start = tl.load(cu_num_k_block + batch_idx)
    k_block_end = tl.load(cu_num_k_block + batch_idx + 1)

    for off_k_block in range(k_block_start, k_block_end):
        start_n = tl.load(cu_k_block + off_k_block)
        
        # We only need to enter the calulcation if two conditions are met:
        # 1. the block mask is True
        # 2. causal = False; or when causal = True && the end of the q block is after the start of the k block.
        if tl.load(block_mask + off_head_k * stride_b_nh + off_q_block * stride_b_nq + off_k_block * stride_b_nk) and (not causal or end_m - batch_q_start_idx + offset >= start_n - batch_k_start_idx):
            
            end_n = tl.load(cu_k_block + off_k_block + 1)
            off_n = start_n + tl.arange(0, BLOCK_N)
            
            if EVEN_SEQ_KBLOCK:
                if EVEN_HEADDIM:
                    k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d)
                    v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d)
                else:
                    k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d, 
                                    mask =off_dim[:, None] < headdim, 
                                    other=0.0)
                    v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d, 
                                    mask = off_dim[None, :] < headdim, 
                                    other=0.0)
            else:
                if EVEN_HEADDIM:
                    k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d, 
                                    mask = off_n[None,:] < end_n, 
                                    other=0.0)
                    v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d, 
                                    mask = off_n[:,None] < end_n, 
                                    other=0.0)
                else:
                    k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d, 
                                    mask = (off_n[None,:] < end_n) & (off_dim[:, None] < headdim), 
                                    other=0.0)
                    v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d, 
                                    mask = (off_n[:,None] < end_n) & (off_dim[None, :] < headdim), 
                                    other=0.0)
            
            # Core part: online Softmax
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q_block, k_block, allow_tf32=False) # Provdie allow_tf32=False can achieve better accuracy for float32. 
            qk *= softmax_scale

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            
            if causal:
                qk += tl.where(off_m[:, None] - batch_q_start_idx + offset >= off_n[None, :] - batch_k_start_idx, 0, float('-inf'))
            
            if not EVEN_SEQ_KBLOCK and start_n + BLOCK_N > end_n: 
                qk += tl.where(off_n[None,:] < end_n, 0, float('-inf'))
            
            p = tl.exp(qk)
            
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            
            # Original flashattention here stores and immediately loads, but it seems not necessary in my testing.
            # tl.store(tmp_ptr, alpha, mask = off_m < end_m)
            # alpha = tl.load(tmp_ptr, mask = off_m < end_m)
            # 
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            p = p.to(v.type.element_ty)
            
            
            acc += tl.dot(p, v_block, allow_tf32=False) # Provdie allow_tf32=False can achieve better accuracy for float32. 
            m_i = m_ij

    l_i = tl.where(l_i == 0, 1, l_i) # might be a working trick for the case when l_i is not updated at all. 
    l_recip = 1 / l_i
    # tl.store(tmp_ptr, l_recip, mask = off_m < end_m)
    # l_recip = tl.load(tmp_ptr, mask = off_m < end_m)
    acc = acc * l_recip[:,None]
    acc = acc.to(out.dtype.element_ty)
    
    off_o = off_m[:, None] * stride_o_s + off_head_q * stride_o_h + off_dim[None, :] * stride_o_d
    out_ptr = out + off_o
    if EVEN_SEQ_QBLOCK:
        if EVEN_HEADDIM:
            tl.store(out_ptr, acc)
        else:
            tl.store(out_ptr, acc, mask=off_dim[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptr, acc, mask=off_m[:, None] < end_m)
        else:
            tl.store(out_ptr, acc, mask=(off_m[:, None] < end_m) & (off_dim[None, :] < headdim))

    off_lse = off_head_q * stride_lse_h + off_m * stride_lse_s

    if EVEN_SEQ_QBLOCK:
        tl.store(lse + off_lse, tl.log(l_i))
    else:
        tl.store(lse + off_lse, tl.log(l_i), mask = off_m < end_m)


def _scheduling(block_mask, cu_num_q_block, batch_size, grouping_function):
    num_block_per_group = 1
    
    nhead, num_q_block, num_k_block = block_mask.shape

    # cu_num_q_group[batch_idx] = the start q group index of batch idx
    cu_num_q_group = torch.zeros(
        batch_size + 1,
        device=block_mask.device,
        dtype=torch.int32,
    )
    cu_num_q_group[1:] = torch.ceil((cu_num_q_block[1:] - cu_num_q_block[:-1]) / num_block_per_group).cumsum(dim=0)
    num_q_group = cu_num_q_group[-1]
    
    # q_group_to_batch[group_idx] = the batch id of the group idx
    q_group_to_batch = torch.zeros(
        num_q_group,
        device=block_mask.device,
        dtype=torch.int32,
    )
    q_group_to_batch[cu_num_q_group[1:-1]] = 1
    q_group_to_batch = q_group_to_batch.cumsum(dim=0, dtype=torch.int32)
    
    # q_assignment[head_idx, group_idx, :] = all the q blocks_idx assigned to group_idx for head_idx
    q_assignment = -1 * torch.ones((nhead, num_q_group, num_block_per_group), device=block_mask.device, dtype=torch.int32)

    # a temporary placehoder, just group consecutive block together into a group for now.
    for q_group in range(num_q_group):
        batch_idx = q_group_to_batch[q_group]
        q_block_start_idx = cu_num_q_block[batch_idx]
        q_block_end_idx = cu_num_q_block[batch_idx + 1]
        q_group_index_real = q_group - cu_num_q_group[batch_idx]
        for block in range(num_block_per_group):
            if q_block_start_idx + q_group_index_real * num_block_per_group + block < q_block_end_idx:
                q_assignment[:, q_group, block] = q_block_start_idx + q_group_index_real * num_block_per_group + block

    # k_assignment [head_idx, group_idx, i] = True, means the i-th K block need to be assigned to group_idx (required by some q blocks there)for head_idx
    k_assignment = torch.zeros((nhead, num_q_group, num_k_block), device=block_mask.device, dtype=torch.bool)
    
    for head_idx in range(nhead):
        for q_group in range(num_q_group):
            all_q_blocks_in_group = q_assignment[head_idx, q_group, :]
            for j in all_q_blocks_in_group:
                k_index = torch.where(block_mask[head_idx, j, :])
                print("k_index: ", k_index)
                k_assignment[head_idx, q_group, k_index] = True
                
    return num_block_per_group, num_q_group, cu_num_q_group, q_group_to_batch, q_assignment, k_assignment

def _forward_scheduling(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, grouping_function, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block):
    
    print(f"debug, cu_q_block: {cu_q_block}, cu_num_q_block: {cu_num_q_block}")
    seq_len_q = q.shape[0]
    nhead_q = q.shape[1]
    nhead_k = k.shape[1]
    batch_size = len(cu_q_seqlens) - 1
    assert nhead_q % nhead_k == 0, "nhead_q must be divisible by nhead_k (for GQA)"
    head_q_to_k_ratio = nhead_q // nhead_k

    headdim = q.shape[2]
    assert (q_block_size & (q_block_size - 1) == 0) and (k_block_size & (k_block_size - 1) == 0), "q_block_size and k_block_size must be powers of 2"
    BLOCK_M = q_block_size
    BLOCK_N = k_block_size
    BLOCK_DIM = max(triton.next_power_of_2(headdim), 16)
    
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5
    
    
    out = torch.empty_like(q).contiguous()
    lse = torch.empty((seq_len_q, nhead_q), device=q.device, dtype=torch.float32)
    tmp = torch.empty((seq_len_q, nhead_q), device=q.device, dtype=torch.float32)
    
    EVEN_SEQ_KBLOCK = torch.all((cu_k_seqlens[1:] - cu_k_seqlens[:-1]) % k_block_size == 0).item()
    EVEN_SEQ_QBLOCK = torch.all((cu_q_seqlens[1:] - cu_q_seqlens[:-1]) % q_block_size == 0).item()
    even_headdim = headdim == BLOCK_DIM
    
    num_block_per_group, num_q_group, cu_num_q_group, q_group_to_batch, q_assignment, k_assignment = _scheduling(block_mask, cu_num_q_block, batch_size, grouping_function)
    print(f"num_block_per_group: {num_block_per_group}, num_q_group: {num_q_group}, cu_num_q_group: {cu_num_q_group}, q_group_to_batch: {q_group_to_batch}, q_assignment: {q_assignment}, k_assignment: {k_assignment}")
    # launch kernel
    grid = (num_q_block, nhead_q)

    _fwd_kernel[grid](
        q,
        k,
        v,
        cu_q_seqlens,
        cu_k_seqlens,
        causal,
        softmax_scale,
        cu_q_block, 
        cu_k_block,
        q_block_to_batch,
        cu_num_k_block,
        head_q_to_k_ratio,
        block_mask,
        q_assignment,
        q_group_to_batch,
        cu_num_q_group,
        out,
        lse,
        tmp,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *block_mask.stride(),
        *q_assignment.stride(),
        *out.stride(),
        *lse.stride(),
        headdim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DIM,        
        EVEN_HEADDIM = even_headdim,
        EVEN_SEQ_QBLOCK = EVEN_SEQ_QBLOCK,
        EVEN_SEQ_KBLOCK = EVEN_SEQ_KBLOCK,
    )
    
    return out

