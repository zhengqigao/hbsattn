import torch
from functools import lru_cache

# calculate_blocks is adpated from https://github.com/MoonshotAI/MoBA/blob/master/moba/moba_efficient.py


def calculate_blocks(cu_seqlen: torch.Tensor, block_size: int) -> tuple[int, torch.Tensor, torch.Tensor]:
    """calculate cu_block and block_to_batch"""

    # batch_sizes[batch_idx] = batch size ( seqlen ) of batch idx
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # batch_num_block[batch_idx] = how many block in batch idx
    batch_num_block = (batch_sizes + (block_size - 1)) // block_size
    
    # cu_num_block[batch_idx] = first block id of this batch
    cu_num_block = torch.zeros(
        batch_num_block.numel() + 1,
        device=cu_seqlen.device,
        dtype=batch_num_block.dtype,
    )
    
    
    cu_num_block[1:] = batch_num_block.cumsum(dim=0)
    
    # total block ( for all batch )
    num_block = cu_num_block[-1].item()
    # block_sizes[block_idx] = block_size of block idx
    block_sizes = torch.full(
        (num_block + 1,), block_size, dtype=torch.int32, device=cu_seqlen.device
    )
    
    block_sizes[0] = 0 
    batch_last_block_size = batch_sizes - (batch_num_block - 1) * block_size
    block_sizes[cu_num_block[1:]] = batch_last_block_size
    # cu_block[block_idx] = the start block offset of block idx
    cu_block = block_sizes.cumsum(dim=-1, dtype=torch.int32)
    # block_to_batch[block_idx] = batch idx of the block idx
    block_to_batch = torch.zeros(
        (num_block,), dtype=torch.int32, device=cu_seqlen.device
    )
    block_to_batch[cu_num_block[1:-1]] = 1
    block_to_batch = block_to_batch.cumsum(dim=0, dtype=torch.int32)
    
    return (
        num_block, # the total number of blocks 
        cu_block, # the start and end (sequence) index of each block
        block_to_batch, # block_to_batch[block_idx] represents the batch index of block_idx
        cu_num_block, # cu_num_block[batch_idx+1] - cu_num_block[batch_idx] = batch_num_block[batch_idx]
    )



def base_schedule(num_block_per_group, block_mask, num_q_block, num_q_group, q_group_to_batch, cu_num_q_group, cu_num_q_block):
    '''
    Get the assignment of q blocks to several groups.
    
    Args: 
    num_block_per_group: a >1 integer,the number of blocks per group defined by the user.
    block_mask: a bool tensor of shape (nhead, num_q_block, num_k_block) indicates which q block attends to which k block.
    num_q_block: the sum of total number of q blocks in one batch.
    num_q_group: the sum of total number of q groups in one batch.
    q_group_to_batch: a tensor of shape (num_q_group), indicates the batch index of each q group.
    cu_num_q_group: a tensor of shape (batch_size + 1), indicates the start index of each q group in the batch.
    cu_num_q_block: a tensor of shape (batch_size + 1), indicates the start index of each q block in the batch.
    
    '''
    nhead = block_mask.shape[0]
    q_assignment = num_q_block * torch.ones((nhead, num_q_group, num_block_per_group), device=block_mask.device, dtype=torch.int32)
    for q_group in range(num_q_group):
        batch_idx = q_group_to_batch[q_group]
        q_block_start_idx = cu_num_q_block[batch_idx]
        q_block_end_idx = cu_num_q_block[batch_idx + 1]
        q_group_index_real = q_group - cu_num_q_group[batch_idx]
        for block in range(num_block_per_group):
            if q_block_start_idx + q_group_index_real * num_block_per_group + block < q_block_end_idx:
                q_assignment[:, q_group, block] = q_block_start_idx + q_group_index_real * num_block_per_group + block
    return q_assignment