import torch
# import torch.nn as nn
import torch.nn.functional as F
from line_profiler_pycharm import profile

# This is a collection of custom convolutions implemented in functional form, based on F.conv2d()
# For custom convolutional layers, implemented as nn.modules, see the file CustomLayers.py


@profile
def dir2dir_conv2d(dir_input, para_kernel, diag_kernel, kernel_size):
    # dir_inputs.shape : (N, 4 * in_channels_per_dir, board_size, board_size)
    # para_kernel.shape : (out_channels_per_dir, in_channels_per_dir, kernel_size)
    # diag_kernel.shape : (out_channels_per_dir, in_channels_per_dir, kernel_size)
    half_k = kernel_size // 2
    # Horizontal Convolution Kernel (Expand and pad)
    kernel_h = para_kernel.unsqueeze(-2)
    kernel_h = F.pad(kernel_h, pad=(0, 0, half_k, half_k))
    # Vertical Convolution Kernel (Expand and pad)
    kernel_v = para_kernel.unsqueeze(-1)
    kernel_v = F.pad(kernel_v, pad=(half_k, half_k, 0, 0))
    # Diagonal Convolution Kernel (Using diag_embed)
    kernel_d = torch.diag_embed(diag_kernel)
    # Anti-Diagonal Convolution Kernel (Flip the diagonal kernel)
    kernel_a = torch.flip(kernel_d, dims=[-1])
    # Each kernel_*.shape: (out_channels_per_dir, in_channels_per_dir, kernel_size, kernel_size)
    kernels = torch.cat([kernel_h, kernel_v, kernel_d, kernel_a], dim=0)
    # kernels.shape: (4 * out_channels_per_dir, in_channels_per_dir, kernel_size, kernel_size)
    dir_output = F.conv2d(dir_input, kernels, padding=half_k, groups=4)
    # dir_output.shape : (N, 4 * out_channels_per_dir, board_size, board_size)
    return dir_output


@profile
def dir2point_conv2d(dir_input, para_kernel, diag_kernel, kernel_size):
    # dir_inputs.shape : (N, 4 * in_channels_per_dir, board_size, board_size)
    # para_kernel.shape : (out_channels, in_channels_per_dir, kernel_size)
    # diag_kernel.shape : (out_channels, in_channels_per_dir, kernel_size)
    half_k = kernel_size // 2
    # Horizontal Convolution Kernel (Expand and pad)
    kernel_h = para_kernel.unsqueeze(-2)
    kernel_h = F.pad(kernel_h, pad=(0, 0, half_k, half_k))
    # Vertical Convolution Kernel (Expand and pad)
    kernel_v = para_kernel.unsqueeze(-1)
    kernel_v = F.pad(kernel_v, pad=(half_k, half_k, 0, 0))
    # Diagonal Convolution Kernel (Using diag_embed)
    kernel_d = torch.diag_embed(diag_kernel)
    # Anti-Diagonal Convolution Kernel (Flip the diagonal kernel)
    kernel_a = torch.flip(kernel_d, dims=[-1])
    # Each kernel_*.shape: (out_channels, in_channels_per_dir, kernel_size, kernel_size)
    kernels = torch.cat([kernel_h, kernel_v, kernel_d, kernel_a], dim=1)
    # kernels.shape: (out_channels, 4 * in_channels_per_dir, kernel_size, kernel_size)
    point_output = F.conv2d(dir_input, kernels, padding=half_k, groups=1)
    # dir_output.shape : (N, 4 * out_channels_per_dir, board_size, board_size)
    return point_output


@profile
def point2dir_conv2d(point_input, para_kernel, diag_kernel, kernel_size):
    # point_inputs.shape : (N, in_channels, board_size, board_size)
    # para_kernel.shape : (out_channels_per_dir, in_channels, kernel_size)
    # diag_kernel.shape : (out_channels_per_dir, in_channels, kernel_size)

    dir_output = dir2dir_conv2d(point_input.repeat(1, 4, 1, 1), para_kernel, diag_kernel, kernel_size)
    # dir_output.shape : (N, 4 * out_channels_per_dir, board_size, board_size)
    return dir_output


if __name__ == "__main__":
    num_iter, N = 100, 1000
    in_per_dir = 9
    out_per_dir = 8
    b_size = 15
    k = 3

    dir_inputs = torch.ones((N, 4 * in_per_dir, b_size, b_size), dtype=torch.float32)
    par_kernel = torch.ones((out_per_dir, in_per_dir, k), dtype=torch.float32)
    dia_kernel = torch.ones((out_per_dir, in_per_dir, k), dtype=torch.float32)
    p_inputs = torch.ones((N, in_per_dir, b_size, b_size), dtype=torch.float32)

    for i in range(num_iter):
        print(i)
        dir_outputs = dir2dir_conv2d(dir_inputs, par_kernel, dia_kernel, k)
        p2dir_outputs = point2dir_conv2d(p_inputs, par_kernel, dia_kernel, k)
        dir2p_outputs = dir2point_conv2d(dir_inputs, par_kernel, dia_kernel, k)


#
# class HorizontalConvolution(nn.Module):
#     def __init__(self, input_channel, output_channel, kernel_size, kernel_tensor):
#         super(HorizontalConvolution, self).__init__()
#
#         # Store the input arguments
#         self.input_channel = input_channel
#         self.output_channel = output_channel
#         self.kernel_size = kernel_size
#
#         # The kernel tensor is provided directly as a parameter
#         assert kernel_tensor.shape == (output_channel, input_channel, kernel_size), \
#             "kernel_tensor must have shape (output_channel, input_channel, kernel_size)"
#         self.kernel_tensor = kernel_tensor
#
#     def forward(self, x):
#         # Reshape kernel_tensor to 4D (output_channel, input_channel, 1, kernel_size)
#         kernel = self.kernel_tensor.view(self.output_channel, self.input_channel, 1, self.kernel_size)
#
#         # Apply 2D convolution using the custom kernel
#         # Use F.conv2d, with stride=1, padding=0, and dilation=1 as defaults
#         output = F.conv2d(x, kernel, stride=1, padding=(0, self.kernel_size // 2), dilation=1)
#
#         return output
#
#
# class VerticalConvolution(nn.Module):
#     def __init__(self, input_channel, output_channel, kernel_size, kernel_tensor):
#         super(VerticalConvolution, self).__init__()
#
#         # Store the input arguments
#         self.input_channel = input_channel
#         self.output_channel = output_channel
#         self.kernel_size = kernel_size
#
#         # The kernel tensor is provided directly as a parameter
#         assert kernel_tensor.shape == (output_channel, input_channel, kernel_size), \
#             "kernel_tensor must have shape (output_channel, input_channel, kernel_size)"
#         self.kernel_tensor = kernel_tensor
#
#     def forward(self, x):
#         # Reshape kernel_tensor to 4D (output_channel, input_channel, kernel_size, 1)
#         kernel = self.kernel_tensor.view(self.output_channel, self.input_channel, self.kernel_size, 1)
#
#         # Apply 2D convolution using the custom kernel
#         # Use F.conv2d, with stride=1, padding=0, and dilation=1 as defaults
#         # Padding is applied only to the vertical dimension
#         output = F.conv2d(x, kernel, stride=1, padding=(self.kernel_size // 2, 0), dilation=1)
#
#         return output
#
#
# class DiagonalConvolution(nn.Module):
#     def __init__(self, input_channel, output_channel, kernel_size, kernel_tensor):
#         super(DiagonalConvolution, self).__init__()
#
#         # Store the input arguments
#         self.input_channel = input_channel
#         self.output_channel = output_channel
#         self.kernel_size = kernel_size
#
#         # The kernel tensor is provided directly as a parameter
#         assert kernel_tensor.shape == (output_channel, input_channel, kernel_size), \
#             "kernel_tensor must have shape (output_channel, input_channel, kernel_size)"
#         self.kernel_tensor = kernel_tensor
#
#     def forward(self, x):
#         # Construct the diagonal kernel
#         # kernel_tensor has shape (output_channel, input_channel, kernel_size)
#         # torch.diag_embed creates a 4D kernel of shape (output_channel, input_channel, kernel_size, kernel_size)
#         kernel = torch.diag_embed(self.kernel_tensor)
#
#         # Apply 2D convolution using the diagonal kernel
#         # Use stride=1, padding=(kernel_size // 2, kernel_size // 2), and dilation=1
#         output = F.conv2d(x, kernel, stride=1, padding=(self.kernel_size // 2, self.kernel_size // 2), dilation=1)
#
#         return output
#
#
# class AntiDiagonalConvolution(nn.Module):
#     def __init__(self, input_channel, output_channel, kernel_size, kernel_tensor):
#         super(AntiDiagonalConvolution, self).__init__()
#
#         # Store the input arguments
#         self.input_channel = input_channel
#         self.output_channel = output_channel
#         self.kernel_size = kernel_size
#
#         # The kernel tensor is provided directly as a parameter
#         assert kernel_tensor.shape == (output_channel, input_channel, kernel_size), \
#             "kernel_tensor must have shape (output_channel, input_channel, kernel_size)"
#         self.kernel_tensor = kernel_tensor
#
#     def forward(self, x):
#         # Construct the diagonal kernel
#         # kernel_tensor has shape (output_channel, input_channel, kernel_size)
#         # torch.diag_embed creates a 4D kernel of shape (output_channel, input_channel, kernel_size, kernel_size)
#         kernel = torch.diag_embed(self.kernel_tensor)
#
#         # Flip the spatial dimensions of the kernel to make it anti-diagonal
#         kernel = torch.flip(kernel, dims=[-1])  # Flip along the last dimension (columns)
#
#         # Apply 2D convolution using the anti-diagonal kernel
#         # Use stride=1, padding=(kernel_size // 2, kernel_size // 2), and dilation=1
#         output = F.conv2d(x, kernel, stride=1, padding=(self.kernel_size // 2, self.kernel_size // 2), dilation=1)
#
#         return output
#
#
# class DirectionalConvolution(nn.Module):
#     def __init__(self, input_channels_per_direction, output_channels_per_direction, kernel_size, kernel_tensor):
#         super(DirectionalConvolution, self).__init__()
#
#         # Store parameters
#         self.input_channels_per_direction = input_channels_per_direction
#         self.output_channels_per_direction = output_channels_per_direction
#         self.kernel_size = kernel_size
#
#         assert kernel_tensor.shape == (output_channels_per_direction, input_channels_per_direction, kernel_size), \
#             "kernel_tensor must have shape (output_channels_per_direction, input_channels_per_direction, kernel_size)"
#         self.kernel_tensor = kernel_tensor
#
#     def forward(self, x):
#         # 1. Horizontal Convolution Kernel (Expand and pad)
#         kernel_h = self.kernel_tensor.unsqueeze(-2)  # Add a new dimension for rows (Shape: (out_channels, in_channels, 1, kernel_size))
#         kernel_h = F.pad(kernel_h, pad=(0, 0, self.kernel_size // 2, self.kernel_size // 2))  # Pad along rows (Shape: (out_channels, in_channels, kernel_size, kernel_size))
#
#         # 2. Vertical Convolution Kernel (Expand and pad)
#         kernel_v = self.kernel_tensor.unsqueeze(-1)  # Add a new dimension for columns (Shape: (out_channels, in_channels, kernel_size, 1))
#         kernel_v = F.pad(kernel_v, pad=(self.kernel_size // 2, self.kernel_size // 2, 0, 0))  # Pad along columns (Shape: (out_channels, in_channels, kernel_size, kernel_size))
#
#         # 3. Diagonal Convolution Kernel (Using diag_embed)
#         kernel_d = torch.diag_embed(self.kernel_tensor)  # Diagonal kernel (Shape: (out_channels, in_channels, kernel_size, kernel_size))
#
#         # 4. Anti-Diagonal Convolution Kernel (Flip the diagonal kernel)
#         kernel_ad = torch.flip(kernel_d, dims=[-1])  # Flip along the spatial columns dimension
#
#         # Stack the kernels together
#         kernels = torch.cat([kernel_h, kernel_v, kernel_d, kernel_ad], dim=0)
#
#         # Perform the convolution using groups=4
#         # Each of the 4 kernels will be applied to its corresponding slice of the input tensor
#         output = F.conv2d(x, kernels, stride=1, padding=(self.kernel_size // 2, self.kernel_size // 2), groups=4)
#
#         return output
#
#
# # Assuming the DirectionalConvolution class has been defined as provided above
#
# # Test parameters
# input_channels_per_direction = 1
# output_channels_per_direction = 1
# kernel_size = 5
# output_spatial_dim = (6, 10)  # (height, width)
#
# # Calculate the input spatial dimensions based on the output dimensions and kernel size (no stride or dilation)
# input_spatial_dim = (output_spatial_dim[0], output_spatial_dim[1])
#
# # Create a random input tensor
# batch_size = 1
# input_channels = 4 * input_channels_per_direction  # 4 directions
# input_tensor = torch.randn(batch_size, input_channels, *input_spatial_dim)
#
# # Define the kernel_tensor for the test
# # Shape: (output_channels_per_direction, input_channels_per_direction, kernel_size)
# kernel_tensor = torch.randn(output_channels_per_direction, input_channels_per_direction, kernel_size)
#
# # Instantiate the DirectionalConvolution layer
# directional_conv = DirectionalConvolution(
#     input_channels_per_direction=input_channels_per_direction,
#     output_channels_per_direction=output_channels_per_direction,
#     kernel_size=kernel_size,
#     kernel_tensor=nn.Parameter(kernel_tensor)  # Pass as trainable parameter
# )
#
# # Forward pass
# output = directional_conv(input_tensor)
#
# # Print the input and output shapes to verify correctness
# print("Input shape:", input_tensor.shape)
# print("Kernel tensor shape:", kernel_tensor.shape)
# print("Output shape:", output.shape)
#
# # Validate the output spatial dimensions
# assert output.shape == (batch_size, 4 * output_channels_per_direction, *output_spatial_dim), \
#     f"Expected output shape does not match. Got {output.shape}, expected {(batch_size, 4 * output_channels_per_direction, *output_spatial_dim)}"
#
# print("DirectionalConvolution test passed!")
#
#
#
# # class ExplicitGroupedPointwiseConvolutionWithExtraChannels(nn.Module):
# #     def __init__(self, input_channels, output_channels):
# #         super(ExplicitGroupedPointwiseConvolutionWithExtraChannels, self).__init__()
# #
# #         # Fixed number of groups = 4
# #         assert (input_channels - 3) % 4 == 0, "The number of input channels minus 3 must be divisible by 4"
# #         assert output_channels % 4 == 0, "The number of output channels must be divisible by 4"
# #
# #         self.num_groups = 4
# #         self.input_channels_per_group = (input_channels - 3) // 4  # Divide input channels (excluding extra 3) by groups
# #         self.output_channels_per_group = output_channels // 4  # Divide output channels by groups
# #
# #         # Define the shared trainable transformation matrix A
# #         self.conv = nn.Conv2d(
# #             in_channels=self.input_channels_per_group + 3,  # Group input channels + 3 shared channels
# #             out_channels=self.output_channels_per_group,  # Each group produces output_channels_per_group
# #             kernel_size=1,  # Pointwise convolution
# #             bias=False  # No bias, since we'll use batch normalization
# #         )
# #
# #         # Separate batch normalization layers for each group (explicit, no loop)
# #         self.bn1 = nn.BatchNorm2d(self.output_channels_per_group)  # BatchNorm for group 1
# #         self.bn2 = nn.BatchNorm2d(self.output_channels_per_group)  # BatchNorm for group 2
# #         self.bn3 = nn.BatchNorm2d(self.output_channels_per_group)  # BatchNorm for group 3
# #         self.bn4 = nn.BatchNorm2d(self.output_channels_per_group)  # BatchNorm for group 4
# #
# #     def forward(self, x):
# #         batch_size, channels, height, width = x.shape
# #         assert channels == self.num_groups * self.input_channels_per_group + 3, \
# #             f"Input channels must be {self.num_groups * self.input_channels_per_group + 3}"
# #
# #         # Generalized slicing of input into 4 groups
# #         main_groups = x[:, :self.num_groups * self.input_channels_per_group, :, :]  # Main group channels
# #         extra_channels = x[:, self.num_groups * self.input_channels_per_group:, :, :]  # Last 3 channels
# #
# #         group1 = main_groups[:, 0:self.input_channels_per_group, :, :]  # First group
# #         group2 = main_groups[:, self.input_channels_per_group:2 * self.input_channels_per_group, :, :]  # Second group
# #         group3 = main_groups[:, 2 * self.input_channels_per_group:3 * self.input_channels_per_group, :,
# #                  :]  # Third group
# #         group4 = main_groups[:, 3 * self.input_channels_per_group:, :, :]  # Fourth group
# #
# #         # Concatenate the additional channels with each group
# #         group1 = torch.cat([group1, extra_channels], dim=1)
# #         group2 = torch.cat([group2, extra_channels], dim=1)
# #         group3 = torch.cat([group3, extra_channels], dim=1)
# #         group4 = torch.cat([group4, extra_channels], dim=1)
# #
# #         # Apply the shared convolution to each group
# #         out1 = self.conv(group1)
# #         out2 = self.conv(group2)
# #         out3 = self.conv(group3)
# #         out4 = self.conv(group4)
# #
# #         # Apply separate batch normalization to each group
# #         out1 = self.bn1(out1)
# #         out2 = self.bn2(out2)
# #         out3 = self.bn3(out3)
# #         out4 = self.bn4(out4)
# #
# #         # Concatenate the results along the channel dimension
# #         output = torch.cat([out1, out2, out3, out4], dim=1)  # Shape: (batch, total_output_channels, height, width)
# #
# #         return output
# #
# #
# # # Example usage with custom parameters
# # input_channels = 39  # Total number of input channels (36 group channels + 3 additional channels)
# # output_channels = 32  # Total number of output channels (must be divisible by 4)
# #
# # # Create the layer
# # layer = ExplicitGroupedPointwiseConvolutionWithExtraChannels(input_channels, output_channels)
# #
# # # Create a random input tensor (batch_size=16, channels=39, height=15, width=15)
# # input_tensor = torch.randn(16, 39, 15, 15)
# #
# # # Forward pass
# # output = layer(input_tensor)
# # print("Output shape:", output.shape)  # Expected: (16, 32, 15, 15)
# #
# # # # Example training
# # # target = torch.randn(16, 32, 15, 15)  # Target tensor
# # # loss_fn = nn.MSELoss()
# # # optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
# # #
# # # loss = loss_fn(output, target)
# # # loss.backward()
# # # optimizer.step()
# #
# #
# # class ExplicitGroupedPointwiseConvolutionWithSeparateBN(nn.Module):
# #     def __init__(self, input_channels, output_channels):
# #         super(ExplicitGroupedPointwiseConvolutionWithSeparateBN, self).__init__()
# #
# #         # Fixed number of groups = 4
# #         assert input_channels % 4 == 0, "The number of input channels must be divisible by 4"
# #         assert output_channels % 4 == 0, "The number of output channels must be divisible by 4"
# #
# #         self.num_groups = 4
# #         self.input_channels_per_group = input_channels // 4  # Divide input channels by number of groups
# #         self.output_channels_per_group = output_channels // 4  # Divide output channels by number of groups
# #
# #         # Define the shared trainable transformation matrix A
# #         self.conv = nn.Conv2d(
# #             in_channels=self.input_channels_per_group,  # Each group has input_channels_per_group
# #             out_channels=self.output_channels_per_group,  # Each group produces output_channels_per_group
# #             kernel_size=1,  # Pointwise convolution
# #             bias=False  # No bias, since we'll use batch normalization
# #         )
# #
# #         # Separate batch normalization layers for each group (explicit, no loop)
# #         self.bn1 = nn.BatchNorm2d(self.output_channels_per_group)  # BatchNorm for group 1
# #         self.bn2 = nn.BatchNorm2d(self.output_channels_per_group)  # BatchNorm for group 2
# #         self.bn3 = nn.BatchNorm2d(self.output_channels_per_group)  # BatchNorm for group 3
# #         self.bn4 = nn.BatchNorm2d(self.output_channels_per_group)  # BatchNorm for group 4
# #
# #     def forward(self, x):
# #         batch_size, channels, height, width = x.shape
# #         assert channels == self.num_groups * self.input_channels_per_group, \
# #             f"Input channels must be {self.num_groups * self.input_channels_per_group}"
# #
# #         # Generalized slicing of input into 4 groups
# #         group1 = x[:, 0:self.input_channels_per_group, :, :]  # First group
# #         group2 = x[:, self.input_channels_per_group:2 * self.input_channels_per_group, :, :]  # Second group
# #         group3 = x[:, 2 * self.input_channels_per_group:3 * self.input_channels_per_group, :, :]  # Third group
# #         group4 = x[:, 3 * self.input_channels_per_group:, :, :]  # Fourth group
# #
# #         # Apply the shared convolution to each group
# #         out1 = self.conv(group1)
# #         out2 = self.conv(group2)
# #         out3 = self.conv(group3)
# #         out4 = self.conv(group4)
# #
# #         # Apply separate batch normalization to each group
# #         out1 = self.bn1(out1)
# #         out2 = self.bn2(out2)
# #         out3 = self.bn3(out3)
# #         out4 = self.bn4(out4)
# #
# #         # Concatenate the results along the channel dimension
# #         output = torch.cat([out1, out2, out3, out4], dim=1)  # Shape: (batch, total_output_channels, height, width)
# #
# #         return output
#
#
# # import torch
# #
# # # # Input: 3D tensor (batch_size, dim1, dim2)
# # # x = torch.randn(2, 4, 5)  # Shape: (batch_size=2, dim1=4, dim2=5)
# # #
# # # # Create a 4D tensor with diagonals along the last two dimensions
# # # diag_tensor = torch.diag_embed(x)
# # #
# # # print(diag_tensor.shape)  # Output: (2, 4, 5, 5)
# # #
# # # # Define dimensions
# # # batch_size = 2
# # # dim1 = 3
# # # dim2 = 4  # This is the size of the last two dimensions (square matrix)
# #
# # # Define the anti-diagonal values
# # # values = torch.randn(batch_size, dim1, dim2)  # Shape: (batch_size, dim1, dim2)
# # values = torch.tensor([[[1, 2, 3, 4],
# #                         [5, 6, 7, 8],
# #                         [9, 10, 11, 12]],
# #                        [[13, 14, 15, 16],
# #                         [17, 18, 19, 20],
# #                         [21, 22, 23, 24]]])
# #
# # # Create a diagonal tensor along the last two dimensions
# # diag_tensor = torch.diag_embed(values)  # Shape: (batch_size, dim1, dim2, dim2)
# #
# # # Flip along the last dimension to convert to anti-diagonal
# # anti_diag_tensor = torch.flip(diag_tensor, dims=[-1])  # Shape: (batch_size, dim1, dim2, dim2)
# #
# # print(diag_tensor)
# # print(anti_diag_tensor)
# #
# # # import torch
# # # import torch.nn.functional as F
# # #
# # # # Example tensor: encoded (1, 3, 15, 15)
# # # encoded = torch.zeros(1, 3, 15, 15)  # Example tensor
# # # encoded[0, 0, 4:6, 4:6] = 1.0
# # #
# # # # Define the horizontal sliding window kernel (size=5)
# # # horizontal_kernel = torch.ones(3, 1, 1, 5)  # Shape: (3, 1, 1, 5)
# # # # Define the vertical sliding window kernel (size=5)
# # # vertical_kernel = torch.ones(3, 1, 5, 1)  # Shape: (3, 1, 5, 1)
# # # # Apply horizontal convolution
# # # horizontal_sums = F.conv2d(
# # #     encoded, horizontal_kernel, padding=(0, 2), groups=3
# # # )  # Shape: (32, 3, 15, 15)
# # # # Apply vertical convolution
# # # vertical_sums = F.conv2d(
# # #     encoded, vertical_kernel, padding=(2, 0), groups=3
# # # )  # Shape: (32, 3, 15, 15)
# # #
# # # # Define kernels for diagonal sums (5x5)
# # # main_diag_kernel = torch.zeros(3, 1, 5, 5)  # Shape: (3, 1, 5, 5)
# # # anti_diag_kernel = torch.zeros(3, 1, 5, 5)  # Shape: (3, 1, 5, 5)
# # #
# # # # Fill main diagonal (top-left to bottom-right) with 1s
# # # for i in range(5):
# # #     main_diag_kernel[:, :, i, i] = 1  # Set diagonal elements to 1
# # #
# # # # Fill anti-diagonal (top-right to bottom-left) with 1s
# # # for i in range(5):
# # #     anti_diag_kernel[:, :, i, 4 - i] = 1  # Set anti-diagonal elements to 1
# # #
# # # # Apply convolution for main diagonal
# # # main_diag_sums = F.conv2d(
# # #     encoded, main_diag_kernel, padding=(2, 2), groups=3
# # # )  # Shape: (32, 3, 15, 15)
# # #
# # # # Apply convolution for anti-diagonal
# # # anti_diag_sums = F.conv2d(
# # #     encoded, anti_diag_kernel, padding=(2, 2), groups=3
# # # )  # Shape: (32, 3, 15, 15)
# # #
# # # # Concatenate all sums (horizontal, vertical, main diagonal, anti-diagonal)
# # # combined_sums = torch.cat(
# # #     [horizontal_sums, vertical_sums, main_diag_sums, anti_diag_sums], dim=1
# # # )  # Shape: (32, 9, 15, 15)
# # #
# # # print(combined_sums.shape)  # Output: (32, 9, 15, 15)
# #
