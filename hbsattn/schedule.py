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

def base_schedule_backup3(num_block_per_group, block_mask, num_q_block, num_q_group, q_group_to_batch, cu_num_q_group, cu_num_q_block):
    '''
    Get the assignment of q blocks to several groups.
    
    Args: 
    num_block_per_group: a >1 integer, the number of blocks per group defined by the user.
    block_mask: a bool tensor of shape (nhead, num_q_block, num_k_block) indicates which q block attends to which k block.
    num_q_block: the sum of total number of q blocks.
    num_q_group: the sum of total number of q groups.
    q_group_to_batch: a tensor of shape (num_q_group), indicates the batch index of each q group.
    cu_num_q_group: a tensor of shape (batch_size + 1), indicates the start number index of each q group.
    cu_num_q_block: a tensor of shape (batch_size + 1), indicates the start number index of each q block.
    '''
    nhead = block_mask.shape[0]
    device = block_mask.device
    batch_size = len(cu_num_q_group) - 1
    
    # ==================== Fully Vectorized Implementation ====================
    
    # Step 1: Compute per-batch block counts and group counts
    cur_block_num = cu_num_q_block[1:] - cu_num_q_block[:-1]  # [batch_size]
    num_groups_per_batch = (cur_block_num + num_block_per_group - 1) // num_block_per_group  # [batch_size]
    
    # Step 2: Create a flat index tensor for all blocks across all batches
    # We need to create indices like [0,1,2,...,cur_block_num[0]-1, 0,1,2,...,cur_block_num[1]-1, ...]
    # but offset by cu_num_q_block
    
    # Total number of "slots" needed (including padding)
    total_slots = (num_groups_per_batch * num_block_per_group).sum().item()
    
    # Create a mapping from flat index to (batch_idx, within_batch_index)
    # Use torch.repeat_interleave to create batch indices
    divisible_block_nums = num_groups_per_batch * num_block_per_group  # [batch_size]
    batch_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.int32),
        divisible_block_nums
    )  # [total_slots]
    
    # Create within-batch position indices
    within_batch_positions = torch.cat([
        torch.arange(divisible_block_nums[i], device=device, dtype=torch.int32)
        for i in range(batch_size)
    ])  # [total_slots]
    
    # Step 3: Compute the actual q_block indices
    # index = cu_num_q_block[batch_idx] + within_batch_position if valid, else num_q_block
    
    # Get the start indices for each batch
    batch_start_indices = cu_num_q_block[batch_indices]  # [total_slots]
    
    # Get the block counts for each position
    batch_block_counts = cur_block_num[batch_indices]  # [total_slots]
    
    # Create validity mask: within_batch_position < cur_block_num[batch_idx]
    valid_mask = within_batch_positions < batch_block_counts  # [total_slots]
    
    # Compute final indices
    flat_indices = torch.where(
        valid_mask,
        batch_start_indices + within_batch_positions,
        num_q_block  # invalid index
    )  # [total_slots]
    
    # Step 4: Reshape to [num_q_group, num_block_per_group]
    q_assignment_2d = flat_indices.view(num_q_group, num_block_per_group)
    
    # Step 5: Expand to all heads
    q_assignment = q_assignment_2d.unsqueeze(0).expand(nhead, -1, -1)  # [nhead, num_q_group, num_block_per_group]
    
    return q_assignment


if __name__ == "__main__":
    import argparse
    import torch
    from hbsattn.utils import calculate_blocks
    import time 
    parser = argparse.ArgumentParser()
    parser.add_argument('--causal', action='store_true', default=False)
    parser.add_argument('--softmax_scale', type=float, default=None)
    parser.add_argument('--nruns', type=int, default=2)
    parser.add_argument('--nwarmup', type=int, default=1)
    parser.add_argument('--headdim', type=int, default=128)
    parser.add_argument('--unit_block', type=int, default=4, help='the number of blocks per unit, must be divisible by 4')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--sparse_ratio', type=float, default=0.3)
    parser.add_argument('--num_block_per_group', type=int, default=1, help='the number of blocks per group, used only for hbsattn (scheduling mode)')
    parser.add_argument('--block_size', type=int, default=16)
    args = parser.parse_args()
        
    nruns = args.nruns
    nwarmup = args.nwarmup
    causal = args.causal
    softmax_scale = args.softmax_scale
    headdim = args.headdim
    unit_block = args.unit_block
    nhead_k = args.nheads
    nhead_q = args.nheads
    num_block_per_group = args.num_block_per_group
    batch_size = args.batch_size
    q_block_size = args.block_size
    k_block_size = args.block_size
    
    device = torch.cuda.current_device()
    dtype = torch.bfloat16

    cu_k_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * unit_block * q_block_size
    max_k_seqlen = int((cu_k_seqlens[1:] - cu_k_seqlens[:-1]).max().item())
    k_seqlen = cu_k_seqlens[-1].item()
    
    cu_q_seqlens = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * unit_block * k_block_size
    max_q_seqlen = int((cu_q_seqlens[1:] - cu_q_seqlens[:-1]).max().item())
    q_seqlen = cu_q_seqlens[-1].item()
    
    q = torch.randn(q_seqlen, nhead_q, headdim, device=device, dtype=dtype)
    k = torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)
    v =  torch.randn(k_seqlen, nhead_k, headdim, device=device, dtype=dtype)

    # the following information is needed for our HBSAttention implementation. You don't need to change that, providing cu_q/k_seqlens and q/k_blocksize is enough.
    num_q_block, cu_q_block, q_block_to_batch, cu_num_q_block = calculate_blocks(cu_q_seqlens, q_block_size)
    num_k_block, cu_k_block, k_block_to_batch, cu_num_k_block = calculate_blocks(cu_k_seqlens, k_block_size)

    
    block_mask = torch.zeros(nhead_k, num_q_block, num_k_block, device=device, dtype=torch.bool)
    for i in range(num_q_block):
        remainder = i % 4
        wrk = i // 4
        if remainder == 0:
            block_mask[:,i,wrk * 4] = True
        elif remainder == 1:
            block_mask[:,i,wrk * 4 + 0] = True
            block_mask[:,i,wrk * 4 + 2] = True
        if remainder == 2:
            block_mask[:,i,wrk * 4 + 1] = True
        elif remainder == 3:
            block_mask[:,i,wrk * 4 + 1] = True
            block_mask[:,i,wrk * 4 + 3] = True
  
    num_q_group = num_q_block // num_block_per_group
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
    
    start_time = time.time()
    base_schedule(num_block_per_group, block_mask, num_q_block, num_q_group, q_group_to_batch, cu_num_q_group, cu_num_q_block)
    end_time = time.time()
    print(f"base_schedule time: {end_time - start_time:.3e} sec")

    start_time = time.time()
    base_schedule_backup(num_block_per_group, block_mask, num_q_block, num_q_group, q_group_to_batch, cu_num_q_group, cu_num_q_block)
    end_time = time.time()
    print(f"base_schedule_backup time: {end_time - start_time:.3e} sec")

    start_time = time.time()
    base_schedule_backup2(num_block_per_group, block_mask, num_q_block, num_q_group, q_group_to_batch, cu_num_q_group, cu_num_q_block)
    end_time = time.time()
    print(f"base_schedule_backup2 time: {end_time - start_time:.3e} sec")
    
    start_time = time.time()
    base_schedule_backup3(num_block_per_group, block_mask, num_q_block, num_q_group, q_group_to_batch, cu_num_q_group, cu_num_q_block)
    end_time = time.time()
    print(f"base_schedule_backup3 time: {end_time - start_time:.3e} sec")