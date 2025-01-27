import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_convolutions import dir2dir_conv2d, dir2point_conv2d, point2dir_conv2d


class Dir2DirConv2D(nn.Module):
    def __init__(self, in_channels_per_dir, out_channels_per_dir,
                 kernel_size, para_kernel_init=None, diag_kernel_init=None):
        super().__init__()
        self.in_channels_per_dir = in_channels_per_dir
        self.out_channels_per_dir = out_channels_per_dir
        self.kernel_size = kernel_size

        # Initialize parameter tensors for para_kernel and diag_kernel
        self.para_kernel = nn.Parameter(torch.Tensor(out_channels_per_dir, in_channels_per_dir, kernel_size))
        self.diag_kernel = nn.Parameter(torch.Tensor(out_channels_per_dir, in_channels_per_dir, kernel_size))

        self.reset_parameters(para_kernel_init, diag_kernel_init)

    def reset_parameters(self, para_kernel_init=None, diag_kernel_init=None):
        if para_kernel_init is not None:
            self.para_kernel.data.copy_(para_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.para_kernel, a=5**0.5)

        if diag_kernel_init is not None:
            self.diag_kernel.data.copy_(diag_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.diag_kernel, a=5**0.5)

    def forward(self, dir_input):
        return dir2dir_conv2d(dir_input, self.para_kernel, self.diag_kernel, self.kernel_size)


class Dir2PointConv2D(nn.Module):
    def __init__(self, in_channels_per_dir, out_channels, kernel_size, para_kernel_init=None, diag_kernel_init=None):
        super(Dir2PointConv2D, self).__init__()
        self.in_channels_per_dir = in_channels_per_dir
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Initialize parameter tensors for para_kernel and diag_kernel
        self.para_kernel = nn.Parameter(torch.Tensor(out_channels, in_channels_per_dir, kernel_size))
        self.diag_kernel = nn.Parameter(torch.Tensor(out_channels, in_channels_per_dir, kernel_size))

        self.reset_parameters(para_kernel_init, diag_kernel_init)

    def reset_parameters(self, para_kernel_init=None, diag_kernel_init=None):
        if para_kernel_init is not None:
            self.para_kernel.data.copy_(para_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.para_kernel, a=5**0.5)

        if diag_kernel_init is not None:
            # TODO: Change into this safer code throughout the reset_parameters() methods in all custom layers !!!
            # with torch.no_grad():
            #     self.diag_kernel.copy_(diag_kernel_init)

            self.diag_kernel.data.copy_(diag_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.diag_kernel, a=5**0.5)
        return

    def forward(self, dir_input):
        return dir2point_conv2d(dir_input, self.para_kernel, self.diag_kernel, self.kernel_size)


class Point2DirConv2D(nn.Module):
    def __init__(self, in_channels, out_channels_per_dir, kernel_size, para_kernel_init=None, diag_kernel_init=None):
        super(Point2DirConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels_per_dir = out_channels_per_dir
        self.kernel_size = kernel_size

        # Initialize parameter tensors for para_kernel and diag_kernel
        self.para_kernel = nn.Parameter(torch.Tensor(out_channels_per_dir, in_channels, kernel_size))
        self.diag_kernel = nn.Parameter(torch.Tensor(out_channels_per_dir, in_channels, kernel_size))

        self.reset_parameters(para_kernel_init, diag_kernel_init)

    def reset_parameters(self, para_kernel_init=None, diag_kernel_init=None):
        if para_kernel_init is not None:
            self.para_kernel.data.copy_(para_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.para_kernel, a=5**0.5)

        if diag_kernel_init is not None:
            self.diag_kernel.data.copy_(diag_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.diag_kernel, a=5**0.5)

    def forward(self, point_input):
        return point2dir_conv2d(point_input, self.para_kernel, self.diag_kernel, self.kernel_size)


class Dir2DirSum(nn.Module):
    def __init__(self, in_channels_per_dir, win_length, device='cpu'):
        """
        Custom layer for summing directional features in different directions around all spaces.
        Horizontal features are only summed horizontally, etc.

        Args:
            in_channels_per_dir (int): Number of input channels per direction (same as out_channels_per_direction).
            win_length (int): The size of the kernel (defines the neighborhood window size).
            device (torch.device, optional): The device to place tensors on (default: CPU).
        """
        super().__init__()
        self.in_channels_per_dir = in_channels_per_dir
        self.win_length = win_length
        self.device = device

        # Generate the para and diag kernels using the sum_kernel formula
        self.sum_kernel = (
            torch.ones(in_channels_per_dir, device=self.device)
            .diag_embed()
            .unsqueeze(2)
            .repeat(1, 1, self.win_length)
        )

    def forward(self, dir_input):
        """
        Forward pass of the summation layer.

        Args:
            dir_input (Tensor): Input tensor of shape (N, 4 * in_channels_per_dir, H, W).

        Returns:
            Tensor: Output tensor of shape (N, 4 * in_channels, H, W), where each
            channel group corresponds to the directional summations.
        """
        # Apply the point2dir_conv2d function with the sum_kernel for both para and diag
        dir_sum = dir2dir_conv2d(dir_input, self.sum_kernel, self.sum_kernel, self.win_length)
        return dir_sum


class Point2DirSum(nn.Module):
    def __init__(self, in_channels, win_length, device='cpu'):
        """
        Custom layer for summing board values in different directions around all spaces.

        Args:
            in_channels (int): Number of input channels (same as out_channels_per_direction).
            win_length (int): The size of the kernel (defines the neighborhood window size).
            device (torch.device, optional): The device to place tensors on (default: CPU).
        """
        super().__init__()
        self.in_channels = in_channels
        self.win_length = win_length
        self.device = device

        # Generate the para and diag kernels using the sum_kernel formula
        self.sum_kernel = (
            torch.ones(in_channels, device=self.device)
            .diag_embed()
            .unsqueeze(2)
            .repeat(1, 1, self.win_length)
        )

    def forward(self, point_input):
        """
        Forward pass of the summation layer.

        Args:
            point_input (Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            Tensor: Output tensor of shape (N, 4 * in_channels, H, W), where each
            channel group corresponds to the directional summations.
        """
        # Apply the point2dir_conv2d function with the sum_kernel for both para and diag
        dir_sum = point2dir_conv2d(point_input, self.sum_kernel, self.sum_kernel, self.win_length)
        return dir_sum


class DepthWiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size, channel_multiplier=1, initial_kernel=None, device=None):
        """
        Custom Depth-Wise Convolution Layer with Kaiming Uniform Initialization.

        Args:
            in_channels (int): Number of input channels.
            kernel_size (int): Size of the convolution kernel.
            channel_multiplier (int, optional): Multiplicity factor for output channels (default: 1).
            initial_kernel (Tensor, optional): Tensor for explicit kernel initialization.
                                               Shape: (in_channels * channel_multiplier, 1, kernel_size, kernel_size).
            device (torch.device, optional): Device for the layer and kernel tensor (default: CPU).
        """
        super(DepthWiseConv2D, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.channel_multiplier = channel_multiplier
        self.out_channels = in_channels * channel_multiplier
        self.device = device if device is not None else torch.device("cpu")
        self.groups = in_channels  # Depth-wise convolution requires groups == in_channels

        # Initialize kernel using the helper function
        self.kernel = nn.Parameter(self.initialize_kernel(initial_kernel), requires_grad=True)

    def initialize_kernel(self, initial_kernel):
        """
        Helper method to initialize the kernel using Kaiming Uniform Initialization.

        Args:
            initial_kernel (Tensor, optional): Explicit kernel tensor for initialization.

        Returns:
            Tensor: Initialized kernel tensor.
        """
        if initial_kernel is not None:
            # Ensure the provided kernel has the correct shape
            if initial_kernel.shape != (self.out_channels, 1, self.kernel_size, self.kernel_size):
                raise ValueError(
                    f"Initial kernel must have shape {(self.out_channels, 1, self.kernel_size, self.kernel_size)}, "
                    f"but got {initial_kernel.shape}."
                )
            return initial_kernel.to(self.device)
        else:
            # Kaiming Uniform Initialization for trainable kernels
            kernel = torch.empty(
                self.out_channels, 1, self.kernel_size, self.kernel_size, device=self.device
            )
            nn.init.kaiming_uniform_(kernel, a=5 ** 0.5)  # Gain for ReLU activation
            return kernel

    def forward(self, x):
        """
        Forward pass for depth-wise convolution.

        Args:
            x (Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            Tensor: Output tensor of shape (N, out_channels, H_out, W_out).
        """
        # Perform depth-wise convolution using F.conv2d
        return F.conv2d(x, self.kernel, bias=None, stride=1, padding=self.kernel_size // 2, groups=self.groups)


if __name__ == "__main__":
    # Example input tensors
    dir_inputs = torch.randn(1, 12, 8, 8)  # Batch of 128, 4*3 channels, 8x8 board
    point_inputs = torch.randn(1, 3, 8, 8)  # Batch of 128, 3 channels, 8x8 board

    # Dir2DirConv2D example
    dir2dir_layer = Dir2DirConv2D(in_channels_per_dir=3, out_channels_per_dir=6, kernel_size=3)
    output_dir2dir = dir2dir_layer(dir_inputs)
    print("Dir2Dir Output Shape:", output_dir2dir.shape)

    # Dir2PointConv2D example
    dir2point_layer = Dir2PointConv2D(in_channels_per_dir=3, out_channels=8, kernel_size=3)
    output_dir2point = dir2point_layer(dir_inputs)
    print("Dir2Point Output Shape:", output_dir2point.shape)

    # Point2DirConv2D example
    point2dir_layer = Point2DirConv2D(in_channels=3, out_channels_per_dir=1, kernel_size=3)
    output_point2dir = point2dir_layer(point_inputs)
    print("Point2Dir Output Shape:", output_point2dir.shape)

    # Point2DirSum example
    point_inputs = torch.zeros(1, 2, 7, 7)  # Batch of 1,  channel 2, 7x7 board
    point_inputs[0, 0, 1:3, 3:5] = 1.0
    point_inputs[0, 1, 1:4, 3:6] = 1.0
    print("point_inputs:\n", point_inputs)
    dir_sum_layer = Point2DirSum(in_channels=2, win_length=5)
    dir_sum = dir_sum_layer(point_inputs)
    print("Point2DirSum Output Shape:", dir_sum.shape)
    print("dir_sum:\n", dir_sum)


# ********************************* OLD STUFF ***********************************************
# def soft_characteristic(x, centers, accuracy=0.01):
#     """
#     Computes a soft characteristic encoding of the input tensor relative to the given centers.
#     For each element in the input tensor, a value close to 1 is assigned to the closest center,
#     and values near 0 are assigned elsewhere.
#
#     The sharpness of the soft Dirac delta function is controlled by the `accuracy` parameter, which
#     defines the width of the peak.
#
#     Args:
#         x (torch.Tensor): Input tensor with arbitrary shape, dtype=torch.float32.
#         centers (torch.Tensor): 1D tensor of float32 values representing class centers.
#         accuracy (float): Controls the width of the soft Dirac delta function. Default is 0.01.
#                           Smaller values result in sharper peaks.
#
#     Returns:
#         torch.Tensor: Soft characteristic encoding tensor with shape (*x.shape, len(centers)).
#     """
#     # Compute the distances between each input element and the centers
#     distances = x.unsqueeze(-1) - centers  # Shape: (*x.shape, len(centers))
#     # Apply the soft Dirac delta function with accuracy-based scaling
#     return 1.0 / (1.0 + (distances / accuracy) ** 4)
#
#
# class DirectionalConvolution(nn.Module):
#     def __init__(self, input_channels_per_direction, output_channels_per_direction, kernel_size, kernel_tensor):
#         super(DirectionalConvolution, self).__init__()
#         # TODO: Wait a minute, this only works with predefined kernel_tensor ... Or does it?
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
# def directional_sum(one_hot_board, k: int):
#
#     # line = torch.ones(k, CUDA_device=one_hot_board.CUDA_device)
#     kernel_tensor = torch.ones(3, device=one_hot_board.CUDA_device).diag_embed().unsqueeze(2).repeat(1, 1, k)
#     dir_conv = DirectionalConvolution(3, 3, 5, kernel_tensor)
#     sum_one_hot = dir_conv(one_hot_board.repeat(1, 4, 1, 1))
#     return sum_one_hot
#
#
# class AmoebaInterpreter(nn.Module):
#     def __init__(self, args: dict):
#         super(AmoebaInterpreter, self).__init__()
#         # TODO: This should go to the class Amoeba ...
#
#         self.board_size = args.get('board_size')
#         self.win_length = args.get('win_length')
#         self.device = args.get('CUDA_device')
#
#         self.stones = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device=self.device)
#         kernel_tensor = torch.ones(3, device=self.device).diag_embed().unsqueeze(2).repeat(1, 1, self.win_length)
#         self.dir_conv = DirectionalConvolution(3,
#                                                3,
#                                                self.win_length,
#                                                kernel_tensor)
#         self.char_list = torch.arange(-self.win_length, self.win_length+1,
#                                       device=self.device, dtype=torch.float32)
#
#     def forward(self, state):
#         board = state.view(state.shape[0], self.board_size, -1)
#         point_interpreted = soft_characteristic(board, self.stones).permute(0, 3, 1, 2)
#         # point_interpreted: Tensor(N, 3, board_size, board_size)
#
#         dir_sum = self.dir_conv(point_interpreted.repeat(1, 4, 1, 1))
#         dir_sum = dir_sum.permute(0, 2, 3, 1)
#         dir_sum = dir_sum.reshape(*dir_sum.shape[:-1], 4, 3)
#         diff = dir_sum[..., 1] - dir_sum[..., 2]
#         pen1 = 100.0 * (5 - torch.sum(dir_sum, dim=-1))
#         pen2 = 100.0 * (dir_sum[..., 1] * dir_sum[..., 2])
#         x = diff + pen1 + pen2
#         dir_interpreted = soft_characteristic(x, self.char_list)
#         dir_interpreted = dir_interpreted.permute(0, 3, 4, 1, 2)
#         # dir_interpreted: Tensor(N, 4, 11, board_size, board_size)
#
#         return point_interpreted, dir_interpreted
#
#
# class AmoebaTerminal(nn.Module):
#     def __init__(self, args: dict):
#         super(AmoebaTerminal, self).__init__()
#         # TODO: This should go to the class Amoeba ...
#
#         self.interpreter = AmoebaInterpreter(args)
#
#     def forward(self, state):
#         point_interpreted, dir_interpreted = self.interpreter(state)
#
#         plus_signal = torch.amax(dir_interpreted[:, :, -1, :, :], dim=(1, 2, 3)) > 0.5
#         minus_signal = torch.amax(dir_interpreted[:, :, 0, :, :], dim=(1, 2, 3)) > 0.5
#         draw_signal = torch.amax(dir_interpreted, dim=(1, 2, 3, 4)) < 0.5
#
#         return plus_signal, minus_signal, draw_signal
#
#
# class AmoebaEncoder(nn.Module):
#     def __init__(self, args: dict):
#         super(AmoebaEncoder, self).__init__()
#         # TODO: This is actually part of the core model ...
#
#     def forward(self, point_interpreted, dir_interpreted, combine=False):
#         # point_interpreted: Tensor(N, 3, board_size, board_size)
#         # dir_interpreted: Tensor(N, 4, 11, board_size, board_size)
#         point_encoded = point_interpreted
#         # Disregard +5 and -5 lines, as those lead to terminated positions ...
#         dir_encoded = dir_interpreted[:, :, 1: -1, :, :]
#         # point1 = point_encoded.unsqueeze(1)
#         if combine:
#             dir_encoded = torch.cat([point_encoded.unsqueeze(1).repeat(1, 4, 1, 1, 1), dir_encoded], dim=2)
#             # merged_tensor = torch.cat([tensor1, tensor2], dim=1)
#         # reshape by stacking the 4 directions into a grouped channel dimension ...
#         new_shape = (dir_encoded.shape[0], dir_encoded.shape[1] * dir_encoded.shape[2],
#                      dir_encoded.shape[3], dir_encoded.shape[4])
#         dir_encoded = dir_encoded.reshape(new_shape)
#
#         return point_encoded, dir_encoded
