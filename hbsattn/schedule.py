import torch 

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