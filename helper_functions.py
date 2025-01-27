import torch
from line_profiler_pycharm import profile


def soft_characteristic(x, centers, accuracy=0.01):
    """
    Computes a soft characteristic encoding of the input tensor relative to the given centers.
    For each element in the input tensor, a value close to 1 is assigned to the closest center,
    and values near 0 are assigned elsewhere.

    The sharpness of the soft Dirac delta function is controlled by the `accuracy` parameter, which
    defines the width of the peak.

    Args:
        x (torch.Tensor): Input tensor with arbitrary shape, dtype=torch.float32.
        centers (torch.Tensor): 1D tensor of float32 values representing class centers.
        accuracy (float): Controls the width of the soft Dirac delta function. Default is 0.01.
                          Smaller values result in sharper peaks.

    Returns:
        torch.Tensor: Soft characteristic encoding tensor with shape (*x.shape, len(centers)).
    """
    # Compute the distances between each input element and the centers
    distances = x.unsqueeze(-1) - centers  # Shape: (*x.shape, len(centers))
    # Apply the soft Dirac delta function with accuracy-based scaling
    return 1.0 / (1.0 + (distances / accuracy) ** 4)


@profile
def helper_unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    # uniq, inverse = torch.unique(
    #     x, sorted=True, return_inverse=True, dim=dim)
    # perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
    #                     CUDA_device=inverse.CUDA_device)
    # # inverse, perm = inverse.flip([0]), perm.flip([0])
    # return uniq, inverse.new_empty(uniq.size(0)).scatter_(0, inverse, perm)

    # Find unique values and their counts
    # unique_tables, inverse_indices, counts = torch.unique(tables, return_inverse=True, return_counts=True)

    uniq, inverse, count = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim, return_counts=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    # inverse, perm = inverse.flip([0]), perm.flip([0])
    return uniq, count, inverse.new_empty(uniq.size(0)).scatter_(0, inverse, perm)


if __name__ == "__main__":
    # Example input tensors
    tables = torch.tensor([0, 1, 1, 2, 0, 2, 1, 0], dtype=torch.long)
    nodes = torch.tensor([5, 6, 6, 7, 5, 7, 6, 8], dtype=torch.long)
    values = torch.tensor([10.5, 20.0, 25.0, 30.0, 11.0, 35.0, 22.0, 12.5], dtype=torch.float32)

    # Combine `tables` and `nodes` into a single tensor of shape (N, 2)
    combined = torch.stack([tables, nodes], dim=1)  # Shape: (N, 2)
    unique_combined, count, unique_indices = helper_unique(combined, dim=0)

    # Use these indices to select unique values
    unique_values = values[unique_indices]

    # Output
    print("Original Combined Tensor (tables, nodes):")
    print(combined)

    print("\nUnique Combined Tensor (tables, nodes):")
    print(unique_combined)

    print("\nCounts:")
    print(count)

    print("\nIndices of First Occurrences:")
    print(unique_indices)

    print("\nUnique Values (from `values` tensor):")
    print(unique_values)


@profile
def duplicate_indices(tables: torch.Tensor) -> torch.Tensor:
    """
    Assigns incrementing indices (marks) for duplicate entries in the input tensor.

    Parameters:
    - tables (torch.Tensor): A 1D tensor containing the input values.

    Returns:
    - torch.Tensor: A 1D tensor of the same shape as `tables`, where:
      - Unique values are marked as 0.
      - Duplicates are marked incrementally (0, 1, 2, ...) for each occurrence.
    """
    # Find unique values and their counts
    unique_tables, inverse_indices, counts = torch.unique(tables, return_inverse=True, return_counts=True)

    # Initialize the mark tensor with zeros
    mark = torch.zeros_like(tables, dtype=torch.long)

    # Identify duplicate entries
    dupl_indices = torch.nonzero(counts > 1).flatten()

    # Increment the mark for duplicates
    for idx in dupl_indices:
        mask = inverse_indices == idx
        mark[mask] = torch.arange(mask.sum())

    return mark


if __name__ == "__main__":
    # Example input tensor
    tables = torch.tensor([1, 2, 3, 1, 4, 2, 5, 3, 1, 6], dtype=torch.long)

    # Call the function
    mark = duplicate_indices(tables)

    # Output
    print("Tables:", tables)
    print("Mark:", mark)
