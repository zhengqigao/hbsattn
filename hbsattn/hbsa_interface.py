import torch 
from hbsattn.fwd_triton_fix_tile_size import _forward_fix_tile_size
from hbsattn.fwd_triton_auto_tile_size import _forward_auto_tile_size
from hbsattn.utils import calculate_blocks

class _HBSAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, tile_mode = 'auto', num_q_block = None, cu_q_block = None, q_block_to_batch = None, cu_num_q_block = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None, cu_num_k_block = None):
        '''
        '''
        
        
        if num_q_block is None or cu_q_block is None or q_block_to_batch is None or cu_num_q_block is None:
            num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
        if num_k_block is None or cu_k_block is None or k_block_to_batch is None or cu_num_k_block is None:
            num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)
        print(block_mask.shape)
        print(k.shape)
        assert block_mask.dtype == torch.bool, "block_mask must be a boolean tensor"
        assert block_mask.shape == (k.shape[1], num_q_block, num_k_block), f"block_mask must be a boolean tensor of shape (nheads_k, num_q_block, num_k_block) = ({k.shape[1]}, {num_q_block}, {num_k_block})"
        
        if tile_mode == 'auto': # the kernel will be lanunched with tile size automatically determined.
            return _forward_auto_tile_size(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
        elif tile_mode == 'fix': # the kernel will be lanunched with tile size equal to the `q_block_size` and `k_block_size`
            return _forward_fix_tile_size(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
        else:
            raise ValueError(f"Invalid tile mode: {tile_mode}")


    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("It does not support gradient propagation yet")


HBSAttention = _HBSAttentionFunction.apply