import torch 
from hbsattn.fwd_triton_fix_tile_size import _forward_fix_tile_size
from hbsattn.fwd_triton_auto_tile_size import _forward_auto_tile_size
from hbsattn.fwd_triton_scheduling import _forward_scheduling
from hbsattn.utils import calculate_blocks,caculate_groups

class _HBSAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, tile_mode = 'auto', num_block_per_group = 1, num_q_block = None, cu_q_block = None, q_block_to_batch = None, cu_num_q_block = None, num_k_block = None, cu_k_block = None, k_block_to_batch = None, cu_num_k_block = None, num_q_group = None, cu_num_q_group = None, q_group_to_batch = None):
        '''
        '''
        
        
        if num_q_block is None or cu_q_block is None or q_block_to_batch is None or cu_num_q_block is None:
            num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
        if num_k_block is None or cu_k_block is None or k_block_to_batch is None or cu_num_k_block is None:
            num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)

        assert block_mask.dtype == torch.bool, "block_mask must be a boolean tensor"
        assert block_mask.shape == (k.shape[1], num_q_block, num_k_block), f"block_mask must be a boolean tensor of shape (nheads_k, num_q_block, num_k_block) = ({k.shape[1]}, {num_q_block}, {num_k_block})"
        
        if callable(tile_mode): # the kernel will be lanunched with the grouping of q blocks determind by a scheduling algorithm and `num_block_per_group`
            
            if num_q_group is None or cu_num_q_group is None or q_group_to_batch is None:
                num_q_group, cu_num_q_group, q_group_to_batch = caculate_groups(cu_num_q_block, num_block_per_group)
            
            return _forward_scheduling(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, tile_mode, num_block_per_group, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block, num_q_group, cu_num_q_group, q_group_to_batch)
        
        elif tile_mode == 'auto': # the kernel will be lanunched with tile size (BLOCK_M, BLOCK_N) to be factors of q_block_size and k_block_size
            
            return _forward_auto_tile_size(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
        
        elif tile_mode == 'fix': # the kernel will be lanunched with BLOCK_M = q_block_size, BLOCK_N = k_block_size
            
            return _forward_fix_tile_size(q, k, v, cu_q_seqlens, cu_k_seqlens, block_mask, q_block_size, k_block_size, causal, softmax_scale, num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block, num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block)
        
        else:
            raise ValueError(f"Invalid tile mode: {tile_mode}")


    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("It does not support gradient propagation yet")


HBSAttention = _HBSAttentionFunction.apply