import torch

def helper_unique_with_multiplicity(x, multi):
    # Step 1: Get unique rows and their mapping
    uniq, inverse = torch.unique(x, dim=0, return_inverse=True)

    # Step 2: Get any representative index of each unique row
    perm = torch.arange(inverse.size(0), device=inverse.device)
    indices = torch.empty(uniq.size(0), dtype=torch.long, device=x.device)
    indices.scatter_(0, inverse, perm)

    # Step 3: Accumulate multiplicities
    count = torch.zeros(uniq.size(0), dtype=multi.dtype, device=multi.device)
    count.index_put_((inverse,), multi, accumulate=True)

    return indices, count

if __name__ == "__main__":

    # Sample input: (N, 2) tensor with repeating rows
    x = torch.tensor([
        [1, 2],
        [1, 4],
        [1, 2],  # Duplicate of row 0
        [5, 2],
        [1, 4],  # Duplicate of row 1
        [1, 2]   # Duplicate of row 0
    ])

    # Multiplicity tensor (same shape as x, but 1D)
    multi = torch.tensor([2, 1, 3, 4, 5, 6])  # Assigned arbitrarily

    # Run the function
    indices, count = helper_unique_with_multiplicity(x, multi)

    # Print results
    print("Input x:\n", x)
    print("Multiplicity:\n", multi)
    print("Indices of unique rows:\n", indices)
    print("Accumulated counts:\n", count)
