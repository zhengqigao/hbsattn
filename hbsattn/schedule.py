import torch 

def find_next_integer_multiple(x, y):
    '''
    Find the next integer multiple of y that is greater than or equal to x.
    '''
    
    return (x + y - 1) // y * y

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
    device = block_mask.device
    
    # Initialize with invalid index
    q_assignment = num_q_block * torch.ones((nhead, num_q_group, num_block_per_group), device=device, dtype=torch.int32)
    
    # ==================== Fully Vectorized Implementation ====================
    
    # Step 1: Get batch info for all groups
    # batch_idx[g] = batch index of group g
    batch_idx = q_group_to_batch  # [num_q_group]
    
    # Step 2: Get q_block range for each group's batch
    # q_block_start_idx[g] = start q_block index of the batch that group g belongs to
    q_block_start_idx = cu_num_q_block[batch_idx]  # [num_q_group]
    q_block_end_idx = cu_num_q_block[batch_idx + 1]  # [num_q_group]
    
    # Step 3: Get the relative group index within each batch
    # q_group_index_real[g] = relative index of group g within its batch
    q_group_start_in_batch = cu_num_q_group[batch_idx]  # [num_q_group]
    q_group_index_real = torch.arange(num_q_group, device=device, dtype=torch.int32) - q_group_start_in_batch  # [num_q_group]
    
    # Step 4: Compute all candidate q_block indices
    # For each group g and position b, the candidate q_block index is:
    # q_block_start_idx[g] + q_group_index_real[g] * num_block_per_group + b
    
    # Expand dimensions for broadcasting
    q_block_start_idx_expanded = q_block_start_idx.unsqueeze(1)  # [num_q_group, 1]
    q_group_index_real_expanded = q_group_index_real.unsqueeze(1)  # [num_q_group, 1]
    block_offset = torch.arange(num_block_per_group, device=device, dtype=torch.int32).unsqueeze(0)  # [1, num_block_per_group]
    
    # Compute candidate indices: [num_q_group, num_block_per_group]
    candidate_indices = (
        q_block_start_idx_expanded + 
        q_group_index_real_expanded * num_block_per_group + 
        block_offset
    )
    
    # Step 5: Create validity mask
    # Valid if: candidate_index < q_block_end_idx[group]
    q_block_end_idx_expanded = q_block_end_idx.unsqueeze(1)  # [num_q_group, 1]
    valid_mask = candidate_indices < q_block_end_idx_expanded  # [num_q_group, num_block_per_group]
    
    # Step 6: Apply assignment
    # Use where to keep invalid indices as num_q_block
    final_indices = torch.where(
        valid_mask,
        candidate_indices,
        num_q_block  # invalid index
    )  # [num_q_group, num_block_per_group]
    
    # Step 7: Broadcast to all heads (all heads have same assignment)
    q_assignment = final_indices.unsqueeze(0).expand(nhead, -1, -1)  # [nhead, num_q_group, num_block_per_group]
    
    return q_assignment


def base_schedule_backup(num_block_per_group, block_mask, num_q_block, num_q_group, q_group_to_batch, cu_num_q_group, cu_num_q_block):
    '''
    Get the assignment of q blocks to several groups.
    
    Args: 
    num_block_per_group: a >1 integer,the number of blocks per group defined by the user.
    block_mask: a bool tensor of shape (nhead, num_q_block, num_k_block) indicates which q block attends to which k block.
    num_q_block: the sum of total number of q blocks.
    num_q_group: the sum of total number of q groups.
    q_group_to_batch: a tensor of shape (num_q_group), indicates the batch index of each q group.
    cu_num_q_group: a tensor of shape (batch_size + 1), indicates the start number index of each q group.
    cu_num_q_block: a tensor of shape (batch_size + 1), indicates the start number index of each q block.
    
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

def base_schedule_backup2(num_block_per_group, block_mask, num_q_block, num_q_group, q_group_to_batch, cu_num_q_group, cu_num_q_block):
    nhead = block_mask.shape[0]
    q_assignment = num_q_block * torch.ones((nhead, num_q_group, num_block_per_group), device=block_mask.device, dtype=torch.int32)
    batch_size = len(cu_num_q_group) - 1
    start = 0 
    for batch_idx in range(batch_size):
        cur_block_num = cu_num_q_block[batch_idx + 1] - cu_num_q_block[batch_idx]
        num_group = (cur_block_num + num_block_per_group - 1) // num_block_per_group 
        divisible_block_num = num_group * num_block_per_group
        index = torch.ones(divisible_block_num, device=block_mask.device, dtype=torch.int32) * num_q_block
        index[:cur_block_num] = torch.arange(cur_block_num, device=block_mask.device, dtype=torch.int32) + cu_num_q_block[batch_idx]
        index = index.view(num_group, num_block_per_group)
        q_assignment[:, start:start+num_group, :] = index
        start += num_group
    return q_assignment