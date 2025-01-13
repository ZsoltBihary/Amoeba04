import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomDiagonalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CustomDiagonalLayer, self).__init__()
        self.kernel_size = kernel_size
        # Trainable parameters for the diagonal values
        self.values = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))

    def forward(self, x):
        # Create diagonal kernels
        kernel = torch.diag_embed(self.values)  # Shape: (out_channels, in_channels, kernel_size, kernel_size)

        # Perform convolution
        # Note: padding="same" maintains input size; adjust if needed
        out = F.conv2d(x, kernel, padding=self.kernel_size // 2)
        return out

# Example usage
batch_size = 1
in_channels = 2
out_channels = 5
kernel_size = 3
input_tensor = torch.randn(batch_size, in_channels, 8, 8)

# Create the layer and apply it
layer = CustomDiagonalLayer(in_channels, out_channels, kernel_size)
output = layer(input_tensor)

print("Output shape:", output.shape)
# Output shape: (batch_size, out_channels, 8, 8)


# import torch
#
# # # Input: 3D tensor (batch_size, dim1, dim2)
# # x = torch.randn(2, 4, 5)  # Shape: (batch_size=2, dim1=4, dim2=5)
# #
# # # Create a 4D tensor with diagonals along the last two dimensions
# # diag_tensor = torch.diag_embed(x)
# #
# # print(diag_tensor.shape)  # Output: (2, 4, 5, 5)
# #
# # # Define dimensions
# # batch_size = 2
# # dim1 = 3
# # dim2 = 4  # This is the size of the last two dimensions (square matrix)
#
# # Define the anti-diagonal values
# # values = torch.randn(batch_size, dim1, dim2)  # Shape: (batch_size, dim1, dim2)
# values = torch.tensor([[[1, 2, 3, 4],
#                         [5, 6, 7, 8],
#                         [9, 10, 11, 12]],
#                        [[13, 14, 15, 16],
#                         [17, 18, 19, 20],
#                         [21, 22, 23, 24]]])
#
# # Create a diagonal tensor along the last two dimensions
# diag_tensor = torch.diag_embed(values)  # Shape: (batch_size, dim1, dim2, dim2)
#
# # Flip along the last dimension to convert to anti-diagonal
# anti_diag_tensor = torch.flip(diag_tensor, dims=[-1])  # Shape: (batch_size, dim1, dim2, dim2)
#
# print(diag_tensor)
# print(anti_diag_tensor)
#
# # import torch
# # import torch.nn.functional as F
# #
# # # Example tensor: encoded (1, 3, 15, 15)
# # encoded = torch.zeros(1, 3, 15, 15)  # Example tensor
# # encoded[0, 0, 4:6, 4:6] = 1.0
# #
# # # Define the horizontal sliding window kernel (size=5)
# # horizontal_kernel = torch.ones(3, 1, 1, 5)  # Shape: (3, 1, 1, 5)
# # # Define the vertical sliding window kernel (size=5)
# # vertical_kernel = torch.ones(3, 1, 5, 1)  # Shape: (3, 1, 5, 1)
# # # Apply horizontal convolution
# # horizontal_sums = F.conv2d(
# #     encoded, horizontal_kernel, padding=(0, 2), groups=3
# # )  # Shape: (32, 3, 15, 15)
# # # Apply vertical convolution
# # vertical_sums = F.conv2d(
# #     encoded, vertical_kernel, padding=(2, 0), groups=3
# # )  # Shape: (32, 3, 15, 15)
# #
# # # Define kernels for diagonal sums (5x5)
# # main_diag_kernel = torch.zeros(3, 1, 5, 5)  # Shape: (3, 1, 5, 5)
# # anti_diag_kernel = torch.zeros(3, 1, 5, 5)  # Shape: (3, 1, 5, 5)
# #
# # # Fill main diagonal (top-left to bottom-right) with 1s
# # for i in range(5):
# #     main_diag_kernel[:, :, i, i] = 1  # Set diagonal elements to 1
# #
# # # Fill anti-diagonal (top-right to bottom-left) with 1s
# # for i in range(5):
# #     anti_diag_kernel[:, :, i, 4 - i] = 1  # Set anti-diagonal elements to 1
# #
# # # Apply convolution for main diagonal
# # main_diag_sums = F.conv2d(
# #     encoded, main_diag_kernel, padding=(2, 2), groups=3
# # )  # Shape: (32, 3, 15, 15)
# #
# # # Apply convolution for anti-diagonal
# # anti_diag_sums = F.conv2d(
# #     encoded, anti_diag_kernel, padding=(2, 2), groups=3
# # )  # Shape: (32, 3, 15, 15)
# #
# # # Concatenate all sums (horizontal, vertical, main diagonal, anti-diagonal)
# # combined_sums = torch.cat(
# #     [horizontal_sums, vertical_sums, main_diag_sums, anti_diag_sums], dim=1
# # )  # Shape: (32, 9, 15, 15)
# #
# # print(combined_sums.shape)  # Output: (32, 9, 15, 15)
#
