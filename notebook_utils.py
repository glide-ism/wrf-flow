import torch

def block_mean_tensor_vectorized(tensor, c):
    """
    Vectorized version of block_mean_tensor for better performance.

    Args:
        tensor: Input tensor of shape (B, 1, n, n) where n is a power of 2
        c: Block factor (power of 2). Each block will be of size (n/c, n/c)

    Returns:
        Tensor of same shape (B, 1, n, n) where each (n/c, n/c) block is filled
        with the mean of that block, robust to NaNs.
    """
    B, _, n, n_check = tensor.shape
    assert n == n_check, f"Expected square matrices, got {n}x{n_check}"
    assert n & (n - 1) == 0, f"n={n} must be a power of 2"
    assert c & (c - 1) == 0, f"c={c} must be a power of 2"
    assert n % c == 0, f"n={n} must be divisible by c={c}"

    block_size = n // c

    # Reshape to separate blocks: (B, 1, c, block_size, c, block_size)
    reshaped = tensor.view(B, 1, c, block_size, c, block_size)

    # Rearrange to group block dimensions: (B, 1, c, c, block_size, block_size)
    blocks = reshaped.permute(0, 1, 2, 4, 3, 5)

    # Compute mean over the last two dimensions (within each block)
    # nanmean handles NaNs robustly
    block_means = torch.nanmean(blocks, dim=(-2, -1), keepdim=True)  # (B, 1, c, c, 1, 1)

    # Expand back to full block size
    expanded_means = block_means.expand(-1, -1, -1, -1, block_size, block_size)

    # Reshape back to original tensor shape
    result = expanded_means.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, 1, n, n)

    return result


