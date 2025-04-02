import torch
import torch.nn.functional as F


def shift_tensor(x, shifts):
    """
    Generalized tensor shifting operation.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, feature, R, C).
        shifts (torch.Tensor): Tensor of shape (num_shifts, 2) specifying relative (row_shift, col_shift).

    Returns:
        torch.Tensor: Shifted tensor of shape (batch, num_shifts, feature, R, C).
    """
    batch, feature, R, C = x.shape
    num_shifts = shifts.shape[0]

    # Determine the max shift for padding
    max_shift_r = shifts[:, 0].abs().max().item()
    max_shift_c = shifts[:, 1].abs().max().item()

    # Apply symmetric padding along both dimensions (R and C)
    x_padded = F.pad(x, (max_shift_c, max_shift_c, max_shift_r, max_shift_r))
    R_pad, C_pad = x_padded.shape[2], x_padded.shape[3]

    # Compute absolute row and column indices after shifting
    base_idx_r = torch.arange(R).unsqueeze(0) + max_shift_r  # (1, R)
    base_idx_c = torch.arange(C).unsqueeze(0) + max_shift_c  # (1, C)

    idx_r = base_idx_r + shifts[:, [0]]  # (num_shifts, R)
    idx_c = base_idx_c + shifts[:, [1]]  # (num_shifts, C)

    # Use advanced indexing to gather the shifted values
    x_shifted = x_padded[:, :, idx_r[:, :, None], idx_c[:, None, :]]  # (batch, feature, num_shifts, R, C)

    # Move shift dimension right after batch
    x_shifted = x_shifted.permute(0, 2, 1, 3, 4)  # (batch, num_shifts, feature, R, C)

    return x_shifted


# Example usage:
batch, feature, R, C = 2, 3, 6, 6
x = 1+torch.arange(batch * feature * R * C).view(batch, feature, R, C).float()

# Define arbitrary shifts [(row_shift, col_shift)]
shifts = torch.tensor([[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]])

x_shifted = shift_tensor(x, shifts)

print("Original x shape:", x.shape)  # (batch, feature, R, C)
print("x_shifted shape:", x_shifted.shape)  # (batch, num_shifts, feature, R, C)

# import torch
# import torch.nn.functional as F
#
# # Example tensor of shape (batch, feature, R, C)
# batch, feature, R, C = 800, 10*8, 15, 15  # Example sizes
# x = 1+torch.arange(batch * feature * R * C).view(batch, feature, R, C).float()
#
# max_shift = 2  # Maximum shift value
# shifts = torch.arange(-max_shift, max_shift + 1)  # [-2, -1, 0, 1, 2]
# num_shifts = len(shifts)
#
# # Apply symmetric padding along both dimensions (R and C)
# x_padded = F.pad(x, (max_shift, max_shift, max_shift, max_shift))  # (batch, feature, R+2*max_shift, C+2*max_shift)
#
# # Create index positions for each shift along both R and C
# idx_r = torch.arange(R).unsqueeze(0) + (max_shift + shifts[:, None])  # (num_shifts, R)
# idx_c = torch.arange(C).unsqueeze(0) + (max_shift + shifts[:, None])  # (num_shifts, C)
#
# # Use advanced indexing to gather the shifted values (correcting row-column ordering)
# x_shifted = x_padded[:, :, idx_r[:, :, None], idx_c[:, None, :]]  # (batch, feature, num_shifts, R, C)
#
# # Permute dimensions to move shift dim right after batch dim
# x_shifted = x_shifted.permute(0, 2, 1, 3, 4)  # (batch, num_shifts, feature, R, C)
#
# print("Original x shape:", x.shape)  # (batch, feature, R, C)
# print("x_shifted shape (5D, reordered):", x_shifted.shape)  # (batch, num_shifts, feature, R, C)
