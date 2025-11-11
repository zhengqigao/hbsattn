import torch
import math 
import triton
import triton.language as tl
import torch.nn.functional as F
from hbsattn.utils import calculate_blocks

__all__ = ['_forward_auto_tile_size']

def get_two_in_factorization(n: int) -> int:
    return (n & -n).bit_length() - 1

def get_autotune_configs(q_block_size, k_block_size):
    """
    Generate autotune configurations based on q_block_size and k_block_size.
    BLOCK_M ranges from 16 to 2^get_two_in_factorization(q_block_size), all powers of 2.
    BLOCK_N ranges from 16 to 2^get_two_in_factorization(k_block_size), all powers of 2.
    """
    configs = []
    
    # Calculate max values
    max_block_m_exp = get_two_in_factorization(q_block_size)
    max_block_n_exp = get_two_in_factorization(k_block_size)
    
    # Minimum is 16 (2^4)
    min_exp = 4
    
    # Generate all powers of 2 from 16 to max
    block_m_values = [2 ** exp for exp in range(min_exp, max_block_m_exp + 1)]
    block_n_values = [2 ** exp for exp in range(min_exp, max_block_n_exp + 1)]
    
    # Generate all combinations
    for block_m in block_m_values:
        for block_n in block_n_values:
            # Adjust num_warps based on block sizes
            if block_m * block_n <= 2048:
                num_warps = 4
            elif block_m * block_n <= 4096:
                num_warps = 8
            else:
                num_warps = 8
            
            configs.append(
                triton.Config(
                    {'BLOCK_M': block_m, 'BLOCK_N': block_n},
                    num_warps=num_warps,
                    num_stages=2
                )
            )
    
    return configs


# BLOCK_M = 16, 32, ..., 2 ** get_two_in_factorization(q_block_size)
# BLOCK_N = 16, 32, ..., 2 ** get_two_in_factorization(k_block_size)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=16, num_stages=1),
        
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    ],
    key=['q_block_size', 'k_block_size']
)
@triton.jit
def _fwd_kernel(
            q, k, v,
            cu_q_seqlens, cu_k_seqlens,
            causal, softmax_scale,
            cu_q_block, cu_k_block,
            q_block_to_batch,
            cu_num_q_block, cu_num_k_block,
            head_q_to_k_ratio,
            block_mask,
            out, lse, tmp,
            stride_q_s, stride_q_h, stride_q_d,
            stride_k_s, stride_k_h, stride_k_d,
            stride_v_s, stride_v_h, stride_v_d,
            stride_b_nh, stride_b_nq, stride_b_nk,
            stride_o_s, stride_o_h, stride_o_d,
            stride_lse_s, stride_lse_h,
            headdim, q_block_size, k_block_size,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_DIM: tl.constexpr,
            EVEN_HEADDIM: tl.constexpr,
            EVEN_SEQ_QBLOCK: tl.constexpr,
            EVEN_SEQ_KBLOCK: tl.constexpr,
        ):
            off_dim = tl.arange(0, BLOCK_DIM)
            
            off_head_q = tl.program_id(2)
            off_head_k = off_head_q // head_q_to_k_ratio
            
            off_q_block = tl.program_id(0)
            off_q_innertile = tl.program_id(1)
            
            num_q_tile_in_block = q_block_size // BLOCK_M
            num_k_tile_in_block = k_block_size // BLOCK_N
            
            batch_idx = tl.load(q_block_to_batch + off_q_block)
            batch_q_start = tl.load(cu_q_seqlens + batch_idx)
            batch_q_end = tl.load(cu_q_seqlens + batch_idx + 1)
            batch_k_start = tl.load(cu_k_seqlens + batch_idx)
            batch_k_end = tl.load(cu_k_seqlens + batch_idx + 1)
            offset = batch_k_end - batch_k_start - (batch_q_end - batch_q_start)
            
            block_q_start = tl.load(cu_q_block + off_q_block)
            start_m = block_q_start + off_q_innertile * BLOCK_M
            if start_m >= batch_q_end:
                return
            end_m = block_q_start + (off_q_innertile + 1) * BLOCK_M
            
            off_m = start_m + tl.arange(0, BLOCK_M)
            q_ptr = q + off_m[:, None] * stride_q_s + off_head_q * stride_q_h + off_dim[None, :] * stride_q_d

            if False: # EVEN_SEQ_QBLOCK             
                if EVEN_HEADDIM:
                    q_block = tl.load(q_ptr)
                else:
                    q_block = tl.load(q_ptr, mask=off_dim[None, :] < headdim, other=0.0)
            else:     
                num_q_tile_in_batch = tl.cdiv(batch_q_end - batch_q_start, BLOCK_M)
                num_q_block_start = tl.load(cu_num_q_block + batch_idx)
                is_last_q_inner_tile_in_batch = (off_q_innertile + (off_q_block - num_q_block_start) * num_q_tile_in_block == num_q_tile_in_batch - 1)
                
                if is_last_q_inner_tile_in_batch:
                    end_m = tl.minimum(end_m, tl.load(cu_q_block + off_q_block + 1))

                if EVEN_HEADDIM:
                    q_block = tl.load(q_ptr, mask=(off_m[:, None] < end_m), other=0.0)
                else:
                    q_block = tl.load(q_ptr, mask=(off_m[:, None] < end_m) & (off_dim[None, :] < headdim), other=0.0)
            
            
            m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=tl.float32)
            
            # tmp_ptr = tmp + off_head_q * stride_lse_h + off_m * stride_lse_s

            num_k_block_start = tl.load(cu_num_k_block + batch_idx)
            num_k_block_end = tl.load(cu_num_k_block + batch_idx + 1)
            
            for off_k_block in range(num_k_block_start, num_k_block_end):
                for off_k_innertile in range(num_k_tile_in_block):
                    block_k_start = tl.load(cu_k_block + off_k_block)
                    start_n = block_k_start + off_k_innertile * BLOCK_N

                    cond1 = start_n < batch_k_end
                    cond2 = tl.load(block_mask + off_head_k * stride_b_nh + off_q_block * stride_b_nq + off_k_block * stride_b_nk)
                    cond3 = (not causal) or (end_m - batch_q_start + offset >= start_n - batch_k_start)
                    if (cond1 and cond2) and cond3:
                        
                        end_n = block_k_start + (off_k_innertile + 1) * BLOCK_N
                        off_n = start_n + tl.arange(0, BLOCK_N)
                        is_last_k_inner_tile_in_batch = False # default to False
                        
                        if False: # EVEN_SEQ_KBLOCK
                            if EVEN_HEADDIM:
                                k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d)
                                v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d)
                            else:
                                k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d, 
                                        mask = off_dim[:, None] < headdim, 
                                        other=0.0)
                                v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d, 
                                        mask = off_dim[None, :] < headdim, 
                                        other=0.0)
                        else:
                            num_k_tile_in_batch = tl.cdiv(batch_k_end - batch_k_start, BLOCK_N)
                            is_last_k_inner_tile_in_batch = (off_k_innertile + (off_k_block - num_k_block_start) * num_k_tile_in_block == num_k_tile_in_batch - 1)

                            if is_last_k_inner_tile_in_batch:
                                end_n = tl.minimum(end_n, tl.load(cu_k_block + off_k_block + 1))
                        
                            if EVEN_HEADDIM:
                                k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d, 
                                        mask = (off_n[None,:] < end_n), 
                                        other=0.0)
                                v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d, 
                                        mask = (off_n[:,None] < end_n), 
                                        other=0.0)
                            else:
                                k_block = tl.load(k + off_n[None,:] * stride_k_s + off_head_k * stride_k_h + off_dim[:, None] * stride_k_d, 
                                        mask = (off_n[None,:] < end_n) & (off_dim[:, None] < headdim), 
                                        other=0.0)
                                v_block = tl.load(v + off_n[:,None] * stride_v_s + off_head_k * stride_v_h + off_dim[None, :] * stride_v_d, 
                                        mask = (off_n[:,None] < end_n) & (off_dim[None, :] < headdim), 
                                        other=0.0)

                        
                        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                        qk += tl.dot(q_block, k_block, allow_tf32=False) # allow_tf32=False
                        qk *= softmax_scale
                        
                        m_ij = tl.maximum(m_i, tl.max(qk, 1))
                        qk -= m_ij[:, None]
                        
                        if causal:
                            qk += tl.where(off_m[:, None] - batch_q_start + offset >= off_n[None, :] - batch_k_start, 0, float('-inf'))
                        
                        if not EVEN_SEQ_KBLOCK and is_last_k_inner_tile_in_batch: 
                            qk += tl.where(off_n[None,:] < end_n, 0, float('-inf'))
                        
                        p = tl.exp(qk)
                        l_ij = tl.sum(p, 1)
                        alpha = tl.exp(m_i - m_ij)
                        
                        l_i = l_i * alpha + l_ij
                        acc = acc * alpha[:, None]
                        p = p.to(v.type.element_ty)
                        
                        acc += tl.dot(p, v_block, allow_tf32=False) #  
                        m_i = m_ij

            l_i = tl.where(l_i == 0, 1, l_i)
            l_recip = 1 / l_i
            acc = acc * l_recip[:,None]
            acc = acc.to(out.dtype.element_ty)
            
            off_o = off_m[:, None] * stride_o_s + off_head_q * stride_o_h + off_dim[None, :] * stride_o_d
            out_ptr = out + off_o
            if False: # EVEN_SEQ_QBLOCK
                if EVEN_HEADDIM:
                    tl.store(out_ptr, acc)
                else:
                    tl.store(out_ptr, acc, mask = off_dim[None, :] < headdim)
            else:
                if EVEN_HEADDIM:
                    tl.store(out_ptr, acc, mask = (off_m[:, None] < end_m))
                else:
                    tl.store(out_ptr, acc, mask = (off_m[:, None] < end_m) & (off_dim[None, :] < headdim))

            off_lse = off_head_q * stride_lse_h + off_m * stride_lse_s
            
            if EVEN_SEQ_QBLOCK:
                tl.store(lse + off_lse, tl.log(l_i))
            else:
                tl.store(lse + off_lse, tl.log(l_i), mask = off_m < end_m)


def _forward_auto_tile_size(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block):
    
    seq_len_q = q.shape[0]
    nhead_q = q.shape[1]
    nhead_k = k.shape[1]
    headdim = q.shape[2]
    
    assert nhead_q % nhead_k == 0, "nhead_q must be divisible by nhead_k (for GQA)"
    head_q_to_k_ratio = nhead_q // nhead_k
    
    # Verify block sizes are valid
    # BLOCK_M = 2 ** get_two_in_factorization(q_block_size) # 2 ** get_two_in_factorization(q_block_size)
    # BLOCK_N = 2 ** get_two_in_factorization(k_block_size)
    # assert BLOCK_M >= 16 and BLOCK_N >= 16, "q_block_size and k_block_size must be integer multiples of 16."
    
    BLOCK_DIM = max(triton.next_power_of_2(headdim), 16)
    softmax_scale = softmax_scale if softmax_scale is not None else headdim ** -0.5
    
    out = torch.empty_like(q).contiguous()
    lse = torch.empty((seq_len_q, nhead_q), device=q.device, dtype=torch.float32)
    tmp = torch.empty((seq_len_q, nhead_q), device=q.device, dtype=torch.float32)
    
    EVEN_SEQ_KBLOCK = torch.all((cu_k_seqlens[1:] - cu_k_seqlens[:-1]) % k_block_size == 0).item()
    EVEN_SEQ_QBLOCK = torch.all((cu_q_seqlens[1:] - cu_q_seqlens[:-1]) % q_block_size == 0).item()
    even_headdim = headdim == BLOCK_DIM
    
    # Grid function uses META to get the autotuned BLOCK_M
    grid = lambda META: (num_q_block, triton.cdiv(q_block_size, META["BLOCK_M"]), nhead_q)
    
    _fwd_kernel[grid](
        q, k, v,
        cu_q_seqlens, cu_k_seqlens,
        causal, softmax_scale,
        cu_q_block, cu_k_block,
        q_block_to_batch,
        cu_num_q_block, cu_num_k_block,
        head_q_to_k_ratio,
        block_mask,
        out, lse, tmp,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *block_mask.stride(),
        *out.stride(),
        *lse.stride(),
        headdim,
        q_block_size,
        k_block_size,
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        BLOCK_DIM=BLOCK_DIM,
        EVEN_HEADDIM=even_headdim,
        EVEN_SEQ_QBLOCK=EVEN_SEQ_QBLOCK,
        EVEN_SEQ_KBLOCK=EVEN_SEQ_KBLOCK,
    )
    
    best_cfg = _fwd_kernel.best_config
    print(f"[Autotune Result] BLOCK_M={best_cfg.kwargs['BLOCK_M']}, BLOCK_N={best_cfg.kwargs['BLOCK_N']}, num_warps={best_cfg.num_warps}, num_stages={best_cfg.num_stages}")

    return out

