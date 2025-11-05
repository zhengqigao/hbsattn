import torch 
from utils.fwd_triton import 

class HBSAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, q_block_size, kv_block_size):
        return _hbsa_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, base_blockmask, max_seqlen_q_, max_seqlen_k_, p_dropout, softmax_scale, is_causal, exact_streaming, return_attn_probs, window_size_left, window_size_right, deterministic)

    @staticmethod
    def backward(ctx, dout):
        raise NotImplementedError("Backward pass not implemented")
