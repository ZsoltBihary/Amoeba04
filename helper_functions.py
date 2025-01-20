import torch
from line_profiler_pycharm import profile

import torch


def soft_one_hot2d(input, num_classes, sharpness=3.0):
    """
    Approximates one-hot encoding using softmax with a sharpness factor,
    with the class dimension placed at the second (channel) index of the output tensor.

    Args:
        input (torch.Tensor): Input tensor (float values), shape (N, H, W).
        num_classes (int): Number of classes for the one-hot encoding.
        sharpness (float): Scaling factor to control the sharpness of the approximation.

    Returns:
        torch.Tensor: Soft one-hot encoded tensor with the class dimension at the second (channel) index.
    """
    # Compute distances to each class index
    distances = torch.abs(input.unsqueeze(-1) - torch.arange(num_classes, device=input.device, dtype=input.dtype))

    # Apply softmax with sharpness scaling
    weights = torch.softmax(-sharpness * distances, dim=-1)

    # Use permute with explicit numeric parameters to move the class dimension to the second index
    # Shape before: (N, H, W, num_classes)
    # We want the output to be (N, num_classes, H, W)
    weights = weights.permute(0, 3, 1, 2)  # (N, num_classes, H, W)

    return weights


if __name__ == "__main__":
    # TODO: Needs testing
    # Example 1: Shape (N, H, W), class dimension at the second (channel) index
    input_tensor_1 = torch.tensor([[[0.5, 2.0], [1.5, 3.0]], [[0.3, 2.3], [1.2, 4.0]]]).float()
    num_classes = 10  # Number of classes
    sharpness = 3.0  # Scaling factor

    # Apply soft one-hot approximation (channel-specific)
    soft_one_hot_1 = soft_one_hot2d(input_tensor_1, num_classes, sharpness)
    print("Soft One-Hot Approximation (Shape: N, C, H, W):")
    print(soft_one_hot_1.shape)  # Expected shape: (N, num_classes, H, W)

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
