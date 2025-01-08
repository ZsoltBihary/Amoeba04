import torch
import torch.nn.functional as F

# Example tensor: encoded (1, 3, 15, 15)
encoded = torch.zeros(1, 3, 15, 15)  # Example tensor
encoded[0, 0, 4:6, 4:6] = 1.0

# Define the horizontal sliding window kernel (size=5)
horizontal_kernel = torch.ones(3, 1, 1, 5)  # Shape: (3, 1, 1, 5)
# Define the vertical sliding window kernel (size=5)
vertical_kernel = torch.ones(3, 1, 5, 1)  # Shape: (3, 1, 5, 1)
# Apply horizontal convolution
horizontal_sums = F.conv2d(
    encoded, horizontal_kernel, padding=(0, 2), groups=3
)  # Shape: (32, 3, 15, 15)
# Apply vertical convolution
vertical_sums = F.conv2d(
    encoded, vertical_kernel, padding=(2, 0), groups=3
)  # Shape: (32, 3, 15, 15)

# Define kernels for diagonal sums (5x5)
main_diag_kernel = torch.zeros(3, 1, 5, 5)  # Shape: (3, 1, 5, 5)
anti_diag_kernel = torch.zeros(3, 1, 5, 5)  # Shape: (3, 1, 5, 5)

# Fill main diagonal (top-left to bottom-right) with 1s
for i in range(5):
    main_diag_kernel[:, :, i, i] = 1  # Set diagonal elements to 1

# Fill anti-diagonal (top-right to bottom-left) with 1s
for i in range(5):
    anti_diag_kernel[:, :, i, 4 - i] = 1  # Set anti-diagonal elements to 1

# Apply convolution for main diagonal
main_diag_sums = F.conv2d(
    encoded, main_diag_kernel, padding=(2, 2), groups=3
)  # Shape: (32, 3, 15, 15)

# Apply convolution for anti-diagonal
anti_diag_sums = F.conv2d(
    encoded, anti_diag_kernel, padding=(2, 2), groups=3
)  # Shape: (32, 3, 15, 15)

# Concatenate all sums (horizontal, vertical, main diagonal, anti-diagonal)
combined_sums = torch.cat(
    [horizontal_sums, vertical_sums, main_diag_sums, anti_diag_sums], dim=1
)  # Shape: (32, 9, 15, 15)

print(combined_sums.shape)  # Output: (32, 9, 15, 15)
