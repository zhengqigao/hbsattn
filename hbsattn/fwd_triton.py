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
    cu_q_block,
    cu_k_block,
    softmax_scale,
    block_mask,
    out,
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
    nhead,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DIM: tl.constexpr
):
    off_head = tl.program_id(1)
    off_q_block = tl.program_id(0)
    off_dim = tl.arange(0, BLOCK_DIM)
    
    start_m = tl.load(cu_q_block + off_q_block)
    end_m = tl.load(cu_q_block + off_q_block + 1)
    off_m = start_m + tl.arange(0, BLOCK_M)
    
    # load the q block
    q_ptr = q + off_m[:, None] * stride_q_s + off_head * stride_q_h + off_dim[None, :] * stride_q_d
    q_block = tl.load(q_ptr, mask=off_m[:, None] < end_m)
    

    # accumulator
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=tl.float32) + 2
    
    # k block loop
    k_block_start = 0
    k_block_end = tl.cdiv(end_m, BLOCK_N)

    for off_k_block in range(k_block_start, k_block_end):
        if tl.load(block_mask + off_head * stride_b_nh + off_q_block * stride_b_nq + off_k_block * stride_b_nk):
            start_n = tl.load(cu_k_block + off_k_block)
            end_n = tl.load(cu_k_block + off_k_block + 1)
            off_n = start_n + tl.arange(0, BLOCK_N)
            k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head * stride_k_h + off_dim[:, None] * stride_k_d, mask = off_n[None,:] < end_n)
            v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head * stride_v_h + off_dim[None, :] * stride_v_d, mask = off_n[:,None] < end_n)
            
            # core part: online Softmax
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q_block, k_block)
            qk *= softmax_scale
            qk += tl.where(off_m[:, None] >= off_n[None, :], 0, float('-inf'))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            p = p.to(v.type.element_ty)
            acc += tl.dot(p, v_block)
            m_i = m_ij
    
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    acc = acc.to(out.dtype.element_ty)
    
    off_o = off_m[:, None] * stride_o_s + off_head * stride_o_h + off_dim[None, :] * stride_o_d
    out_ptr = out + off_o
    tl.store(out_ptr, acc, mask = off_m[:, None] < end_m)


def _forward(q, k, v, cu_seqlens, max_seqlen, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch):
    
    out = torch.empty_like(q).contiguous()
    batch_size = cu_seqlens.shape[0] - 1
    
    seq_len = q.shape[0]
    nhead = q.shape[1]
    headdim = q.shape[2]
    
    BLOCK_M = q_block_size
    BLOCK_N = k_block_size
    BLOCK_DIM = headdim
    
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5
    
    # launch kernel grid according to spliting q. 
    grid = (num_q_block, nhead)

    _fwd_kernel[grid](
        q,
        k,
        v,
        cu_q_block,
        cu_k_block,
        softmax_scale,
        block_mask,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *block_mask.stride(),
        *out.stride(),
        nhead,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DIM
    )
    return out

class _HBSAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block = None, cu_q_block = None, q_block_to_batch = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None):
        '''
        '''
        
        assert causal == True, "causal must be True"
        assert block_mask.dtype == torch.bool, "block_mask must be a boolean tensor"
        
        if num_q_block is None or cu_q_block is None or q_block_to_batch is None:
            num_q_block, cu_q_block, q_block_to_batch = calculate_blocks(cu_seqlens, q_block_size)
        if num_k_block is None or cu_k_block is None or k_block_to_batch is None:
            num_k_block, cu_k_block, k_block_to_batch = calculate_blocks(cu_seqlens, k_block_size)
            
        return _forward(q, k, v, cu_seqlens, max_seqlen, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("It does not support gradient propagation yet")


HBSAttention = _HBSAttentionFunction.apply

if __name__ == "__main__":

    device = torch.cuda.current_device()
    cu_seqlens = torch.tensor([0, 4, 7, 11], dtype=torch.int32, device=device)
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    seqlen = cu_seqlens[-1].item()
    
    nhead = 1
    headdim = 1
    q_block_size = 2 
    k_block_size = 2
    
    min_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).min().item())


    q = torch.randn(seqlen, nhead, headdim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seqlen, nhead, headdim, device=device, dtype=torch.bfloat16)
    v = torch.randn(seqlen, nhead, headdim, device=device, dtype=torch.bfloat16)
    

    num_q_block, cu_q_block, q_block_to_batch = calculate_blocks(cu_seqlens, q_block_size)
    num_k_block, cu_k_block, k_block_to_batch = calculate_blocks(cu_seqlens, k_block_size)


    block_mask = torch.randint(
        low=0,
        high=2,
        size=(nhead, num_q_block, num_k_block),
        dtype=torch.bool,
        device=device
    )

    block_mask = block_mask.contiguous()
    
    out = HBSAttention(q, k, v, cu_seqlens, max_seqlen, block_mask, q_block_size, k_block_size, True, None, num_q_block, cu_q_block, q_block_to_batch, num_k_block, cu_k_block, k_block_to_batch)
    print(out)