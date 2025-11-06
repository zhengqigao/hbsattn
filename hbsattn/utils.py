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



def check_block_mask(block_mask: torch.Tensor, cu_q_blocks: torch.Tensor, q_block_to_batch: torch.Tensor, cu_k_blocks: torch.Tensor, k_block_to_batch: torch.Tensor) -> bool:
    """
    check the validity of a given block mask. 
    Args: 
        block_mask: (nhead, num_q_block, num_k_block)
        cu_q_blocks: (num_q_block + 1)
        cu_k_blocks: (num_k_block + 1)
    Returns:
        bool: True if the block mask is valid, False otherwise.
    """
    assert block_mask.ndim == 3, "block_mask must be a 3D tensor, of the shape (nhead, num_q_block, num_k_block)."
    
    nhead, num_q_block, num_k_block = block_mask.shape
    
    assert max(q_block_to_batch) == max(k_block_to_batch), f"In varlength setting, q_block_to_batch indicates {max(q_block_to_batch)} samples, while k_block_to_batch indicates {max(k_block_to_batch)} samples."
    
    num_sample = max(q_block_to_batch) + 1
    
    # block mask can only 
    
    


if __name__ == "__main__":
    device = torch.cuda.current_device()
    cu_seqlens = torch.tensor([0, 4, 7, 11], dtype=torch.int32, device=device)
    block_size = 2
    num_block, cu_block, block_to_batch = calculate_blocks(cu_seqlens, block_size)
    print(num_block)
    print(cu_block)
    print(block_to_batch)
