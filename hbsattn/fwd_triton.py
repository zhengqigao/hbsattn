import torch
import math 
import triton
import triton.language as tl
import torch.nn.functional as F
from hbsattn.utils import calculate_blocks

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
    stride_o_s,
    stride_o_h,
    stride_o_d,
    stride_lse_s,
    stride_lse_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr
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
    q_block = tl.load(q_ptr, mask=off_m[:, None] < end_m, other=0.0)
    

    # accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=tl.float32)
    
    tmp_ptr = tmp + off_head_q * stride_lse_h + off_m * stride_lse_s
    
    # batch index 
    batch_idx = tl.load(q_block_to_batch + off_q_block)
    
    # get offset and q and k start/end indices in seq
    batch_q_start_idx = tl.load(cu_q_seqlens + batch_idx)
    batch_q_end_idx = tl.load(cu_q_seqlens + batch_idx + 1)
    batch_k_start_idx = tl.load(cu_k_seqlens + batch_idx)
    batch_k_end_idx = tl.load(cu_k_seqlens + batch_idx + 1)
    offset = batch_k_end_idx - batch_q_end_idx
    
    # k block loop, start from the same batch as the q block, and end at the last k block in the same batch.
    k_block_start = tl.load(cu_num_k_block + batch_idx)
    k_block_end = tl.load(cu_num_k_block + batch_idx + 1)

    for off_k_block in range(k_block_start, k_block_end):
        start_n = tl.load(cu_k_block + off_k_block)
        
        # We only need to enter the calulcation if two conditions are met:
        # 1. the block mask is True
        # 2. causal = False; or when causal = True && the end of the q block is after the start of the k block.
        #  and (not causal or end_m - batch_q_start_idx + offset >= start_n - batch_k_start_idx)
        if tl.load(block_mask + off_head_k * stride_b_nh + off_q_block * stride_b_nq + off_k_block * stride_b_nk):
            
            end_n = tl.load(cu_k_block + off_k_block + 1)
            off_n = start_n + tl.arange(0, BLOCK_N)
            
            k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d, mask = off_n[None,:] < end_n, other=0.0)
            v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d, mask = off_n[:,None] < end_n, other=0.0)
            
            # core part: online Softmax
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) ## TODO: in the lask k block, might be a problem because
            qk += tl.dot(q_block, k_block)
            qk *= softmax_scale
            
            if causal:
                qk += tl.where(off_m[:, None] - batch_q_start_idx + offset >= off_n[None, :] - batch_k_start_idx, 0, float('-inf'))
            
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            
            # # # BUG: have to store and immediately load 
            # tl.store(tmp_ptr, alpha, mask = off_m < end_m)
            # alpha = tl.load(tmp_ptr, mask = off_m < end_m)
            
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            p = p.to(v.type.element_ty)
            # tl.device_print("p", p)
            tl.device_print("p type", p.type.element_ty)
            tl.device_print("v_block type", v_block.type.element_ty)
            acc += tl.dot(p, v_block)
            tl.device_print("sum ofacc", acc)
            m_i = m_ij

    # might need to slightly change the code according to the source code given by Flashattention for improved accuracy.
    l_recip = 1 / l_i
    # tl.store(tmp_ptr, l_recip, mask = off_m < end_m)
    # l_recip = tl.load(tmp_ptr, mask = off_m < end_m)
    # tl.device_print("l_recip", l_recip)
    tl.device_print("before acc", acc)
    acc = acc * l_recip[:,None]
    tl.device_print("after acc", acc)
    acc = acc.to(out.dtype.element_ty)
    
    off_o = off_m[:, None] * stride_o_s + off_head_q * stride_o_h + off_dim[None, :] * stride_o_d
    out_ptr = out + off_o
    tl.store(out_ptr, acc, mask = off_m[:, None] < end_m)

    off_lse = off_head_q * stride_lse_h + off_m * stride_lse_s
    tl.store(lse + off_lse, tl.log(l_i), mask = off_m < end_m)

@triton.jit
def _fwd_kernel_backup(
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
    stride_o_s,
    stride_o_h,
    stride_o_d,
    stride_lse_s,
    stride_lse_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr
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
    q_block = tl.load(q_ptr, mask=off_m[:, None] < end_m)
    

    # accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=tl.float32)
    
    tmp_ptr = tmp + off_head_q * stride_lse_h + off_m * stride_lse_s
    
    # batch index 
    batch_idx = tl.load(q_block_to_batch + off_q_block)
    
    # get offset and q and k start/end indices in seq
    batch_q_start_idx = tl.load(cu_q_seqlens + batch_idx)
    batch_q_end_idx = tl.load(cu_q_seqlens + batch_idx + 1)
    batch_k_start_idx = tl.load(cu_k_seqlens + batch_idx)
    batch_k_end_idx = tl.load(cu_k_seqlens + batch_idx + 1)
    offset = batch_k_end_idx - batch_q_end_idx
    
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
            
            k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d, mask = off_n[None,:] < end_n)
            v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d, mask = off_n[:,None] < end_n)
            
            # core part: online Softmax
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q_block, k_block)
            qk *= softmax_scale
            
            if causal:
                qk += tl.where(off_m[:, None] - batch_q_start_idx + offset >= off_n[None, :] - batch_k_start_idx, 0, float('-inf'))
            
            
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            # scale acc_o
            acc_o_scale = tl.exp(m_i - m_ij)

            # BUG: have to store and immediately load
            tl.store(tmp_ptr, acc_o_scale)
            acc_o_scale = tl.load(tmp_ptr)
            acc = acc * acc_o_scale[:, None]
            p = p.to(v.type.element_ty)
            acc += tl.dot(p, v_block)

            # -- update statistics
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)
            
    # might need to slightly change the code according to the source code given by Flashattention for improved accuracy.
    o_scale = tl.exp(m_i - lse_i)
    # # BUG: have to store and immediately load
    tl.store(tmp_ptr, o_scale)
    o_scale = tl.load(tmp_ptr)
    acc = acc * o_scale[:, None]
    acc = acc.to(out.dtype.element_ty)
    
    off_o = off_m[:, None] * stride_o_s + off_head_q * stride_o_h + off_dim[None, :] * stride_o_d
    out_ptr = out + off_o
    tl.store(out_ptr, acc, mask = off_m[:, None] < end_m)

    off_lse = off_head_q * stride_lse_h + off_m * stride_lse_s
    tl.store(lse + off_lse, lse_i, mask = off_m < end_m)


def _forward(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block):
    

    print(f"in _forward, cu_num_q_block: {cu_num_q_block}, cu_num_k_block: {cu_num_k_block}")
    
    seq_len_q = q.shape[0]
    nhead_q = q.shape[1]
    nhead_k = k.shape[1]
    assert nhead_q % nhead_k == 0, "nhead_q must be divisible by nhead_k (for GQA)"
    head_q_to_k_ratio = nhead_q // nhead_k

    headdim = q.shape[2]
    
    BLOCK_M = q_block_size
    BLOCK_N = k_block_size
    BLOCK_DIM = headdim
    
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5
    
    
    out = torch.empty_like(q).contiguous()
    lse = torch.empty((seq_len_q, nhead_q), device=q.device, dtype=torch.float32)
    tmp = torch.empty((seq_len_q, nhead_q), device=q.device, dtype=torch.float32)
    
    # launch kernel grid according to spliting q. 
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
        out,
        lse,
        tmp,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *block_mask.stride(),
        *out.stride(),
        *lse.stride(),
        BLOCK_M,
        BLOCK_N,
        BLOCK_DIM
    )
    return out

class _HBSAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block = None, cu_q_block = None, q_block_to_batch = None, cu_num_q_block = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None, cu_num_k_block = None):
        '''
        '''
        
        assert block_mask.dtype == torch.bool, "block_mask must be a boolean tensor"
        
        if num_q_block is None or cu_q_block is None or q_block_to_batch is None or cu_num_q_block is None:
            num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
        if num_k_block is None or cu_k_block is None or k_block_to_batch is None or cu_num_k_block is None:
            num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)
            
        return _forward(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("It does not support gradient propagation yet")


HBSAttention = _HBSAttentionFunction.apply
