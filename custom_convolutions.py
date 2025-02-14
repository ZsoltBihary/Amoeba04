import torch
# import torch.nn as nn
import torch.nn.functional as F
from line_profiler_pycharm import profile

# This is a collection of custom convolutions implemented in functional form, based on F.conv2d()
# For custom convolutional layers, implemented as nn.modules, see the file CustomLayers.py


def directional_depthwise_conv2d(input_tensor, cen_tensor, dir_tensor):
    """
    Custom depthwise convolution with filters for isotropic features
    and directional filters that are shared across horizontal, vertical, diagonal, and anti-diagonal features.

    Args:
        input_tensor (Tensor): Input tensor of shape (N, C_in, H, W), where C_in = cen_size + 4 * dir_size.
        cen_tensor (Tensor): Kernel tensor for central features, shape (cen_size, cen_k, cen_k).
        dir_tensor (Tensor): Kernel tensor for directional features, shape (dir_size, dir_k).

    Returns:
        output_tensor (Tensor): Output tensor of the same shape as input (N, C_in, H, W).
    """
    N, C, H, W = input_tensor.shape
    cen_size, cen_k, _ = cen_tensor.shape
    dir_size, dir_k = dir_tensor.shape
    assert C == cen_size + 4 * dir_size, "Input channel count must match cen_size + 4*dir_size"
    assert dir_k >= cen_k, "Directional kernel size must not be smaller than central kernel size"
    dir_pad = dir_k // 2
    cen_pad = (dir_k - cen_k) // 2

    # Reshape central kernel to match directional kernel size and conv2d expectations
    cen_kernel = F.pad(cen_tensor, (cen_pad, cen_pad, cen_pad, cen_pad)).unsqueeze(1)

    # Construct directional kernels, all based on dir_tensor
    hor_kernel = F.pad(dir_tensor.unsqueeze(-2), (0, 0, dir_pad, dir_pad)).unsqueeze(1)
    ver_kernel = hor_kernel.transpose(-1, -2)
    dia_kernel = torch.diag_embed(dir_tensor).unsqueeze(1)
    ant_kernel = torch.flip(dia_kernel, dims=[-1])

    # Stack Kernels
    kernels = torch.cat([cen_kernel, hor_kernel, ver_kernel, dia_kernel, ant_kernel], dim=0)

    # Apply Depthwise Convolution
    return F.conv2d(input_tensor, kernels, padding=dir_pad, groups=C)


if __name__ == "__main__":
    # ==== Test the Implementation ====
    N, c_s, d_s, c_k, d_k, board_size = 2, 1, 1, 3, 5, 9

    # Create a random input tensor with grouped channels
    input_tens = torch.zeros(N, c_s + 4 * d_s, board_size, board_size)
    input_tens[0, :, 1, 3] = 1.0
    input_tens[1, :, 0:2, 3:6] = 1.0

    # Define central and directional tensors that will control kernels
    c_tensor = torch.ones(c_s, c_k, c_k)
    d_tensor = torch.ones(d_s, d_k)

    # Apply the custom directional depthwise convolution
    output_tens = directional_depthwise_conv2d(input_tens, c_tensor, d_tensor)
    # output = directional_depthwise_conv2d(input_tensor, central_kernel, para_kernel, diag_kernel, kernel_size)

    print(output_tens.to(dtype=torch.int32))
    # Print output shape
    print("Input Shape:", input_tens.shape)
    print("Output Shape:", output_tens.shape)


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

#
# if __name__ == "__main__":
#     num_iter, N = 100, 1000
#     in_per_dir = 9
#     out_per_dir = 8
#     b_size = 15
#     k = 3
#
#     dir_inputs = torch.ones((N, 4 * in_per_dir, b_size, b_size), dtype=torch.float32)
#     par_kernel = torch.ones((out_per_dir, in_per_dir, k), dtype=torch.float32)
#     dia_kernel = torch.ones((out_per_dir, in_per_dir, k), dtype=torch.float32)
#     p_inputs = torch.ones((N, in_per_dir, b_size, b_size), dtype=torch.float32)
#
#     for i in range(num_iter):
#         print(i)
#         dir_outputs = dir2dir_conv2d(dir_inputs, par_kernel, dia_kernel, k)
#         p2dir_outputs = point2dir_conv2d(p_inputs, par_kernel, dia_kernel, k)
#         dir2p_outputs = dir2point_conv2d(dir_inputs, par_kernel, dia_kernel, k)
#
