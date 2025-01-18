import torch
import torch.nn as nn
import torch.nn.functional as F
# from ClassAmoeba import Amoeba


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


class DirectionalConvolution(nn.Module):
    def __init__(self, input_channels_per_direction, output_channels_per_direction, kernel_size, kernel_tensor):
        super(DirectionalConvolution, self).__init__()
        # TODO: Wait a minute, this only works with predefined kernel_tensor ... Or does it?
        # Store parameters
        self.input_channels_per_direction = input_channels_per_direction
        self.output_channels_per_direction = output_channels_per_direction
        self.kernel_size = kernel_size

        assert kernel_tensor.shape == (output_channels_per_direction, input_channels_per_direction, kernel_size), \
            "kernel_tensor must have shape (output_channels_per_direction, input_channels_per_direction, kernel_size)"
        self.kernel_tensor = kernel_tensor

    def forward(self, x):
        # 1. Horizontal Convolution Kernel (Expand and pad)
        kernel_h = self.kernel_tensor.unsqueeze(-2)  # Add a new dimension for rows (Shape: (out_channels, in_channels, 1, kernel_size))
        kernel_h = F.pad(kernel_h, pad=(0, 0, self.kernel_size // 2, self.kernel_size // 2))  # Pad along rows (Shape: (out_channels, in_channels, kernel_size, kernel_size))

        # 2. Vertical Convolution Kernel (Expand and pad)
        kernel_v = self.kernel_tensor.unsqueeze(-1)  # Add a new dimension for columns (Shape: (out_channels, in_channels, kernel_size, 1))
        kernel_v = F.pad(kernel_v, pad=(self.kernel_size // 2, self.kernel_size // 2, 0, 0))  # Pad along columns (Shape: (out_channels, in_channels, kernel_size, kernel_size))

        # 3. Diagonal Convolution Kernel (Using diag_embed)
        kernel_d = torch.diag_embed(self.kernel_tensor)  # Diagonal kernel (Shape: (out_channels, in_channels, kernel_size, kernel_size))

        # 4. Anti-Diagonal Convolution Kernel (Flip the diagonal kernel)
        kernel_ad = torch.flip(kernel_d, dims=[-1])  # Flip along the spatial columns dimension

        # Stack the kernels together
        kernels = torch.cat([kernel_h, kernel_v, kernel_d, kernel_ad], dim=0)

        # Perform the convolution using groups=4
        # Each of the 4 kernels will be applied to its corresponding slice of the input tensor
        output = F.conv2d(x, kernels, stride=1, padding=(self.kernel_size // 2, self.kernel_size // 2), groups=4)

        return output


def directional_sum(one_hot_board, k: int):

    # line = torch.ones(k, device=one_hot_board.device)
    kernel_tensor = torch.ones(3, device=one_hot_board.device).diag_embed().unsqueeze(2).repeat(1, 1, k)
    dir_conv = DirectionalConvolution(3, 3, 5, kernel_tensor)
    sum_one_hot = dir_conv(one_hot_board.repeat(1, 4, 1, 1))
    return sum_one_hot


class AmoebaInterpreter(nn.Module):
    def __init__(self, args: dict):
        super(AmoebaInterpreter, self).__init__()
        # TODO: This should go to the class Amoeba ...

        self.board_size = args.get('board_size')
        self.win_length = args.get('win_length')
        self.device = args.get('CUDA_device')

        self.stones = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device=self.device)
        kernel_tensor = torch.ones(3, device=self.device).diag_embed().unsqueeze(2).repeat(1, 1, self.win_length)
        self.dir_conv = DirectionalConvolution(3,
                                               3,
                                               self.win_length,
                                               kernel_tensor)
        self.char_list = torch.arange(-self.win_length, self.win_length+1,
                                      device=self.device, dtype=torch.float32)

    def forward(self, state):
        board = state.view(state.shape[0], self.board_size, -1)
        point_interpreted = soft_characteristic(board, self.stones).permute(0, 3, 1, 2)
        # point_interpreted: Tensor(N, 3, board_size, board_size)

        dir_sum = self.dir_conv(point_interpreted.repeat(1, 4, 1, 1))
        dir_sum = dir_sum.permute(0, 2, 3, 1)
        dir_sum = dir_sum.reshape(*dir_sum.shape[:-1], 4, 3)
        diff = dir_sum[..., 1] - dir_sum[..., 2]
        pen1 = 100.0 * (5 - torch.sum(dir_sum, dim=-1))
        pen2 = 100.0 * (dir_sum[..., 1] * dir_sum[..., 2])
        x = diff + pen1 + pen2
        dir_interpreted = soft_characteristic(x, self.char_list)
        dir_interpreted = dir_interpreted.permute(0, 3, 4, 1, 2)
        # dir_interpreted: Tensor(N, 4, 11, board_size, board_size)

        return point_interpreted, dir_interpreted


class AmoebaTerminal(nn.Module):
    def __init__(self, args: dict):
        super(AmoebaTerminal, self).__init__()
        # TODO: This should go to the class Amoeba ...

        self.interpreter = AmoebaInterpreter(args)

    def forward(self, state):
        point_interpreted, dir_interpreted = self.interpreter(state)

        plus_signal = torch.amax(dir_interpreted[:, :, -1, :, :], dim=(1, 2, 3)) > 0.5
        minus_signal = torch.amax(dir_interpreted[:, :, 0, :, :], dim=(1, 2, 3)) > 0.5
        draw_signal = torch.amax(dir_interpreted, dim=(1, 2, 3, 4)) < 0.5

        return plus_signal, minus_signal, draw_signal


class AmoebaEncoder(nn.Module):
    def __init__(self, args: dict):
        super(AmoebaEncoder, self).__init__()
        # TODO: This is actually part of the core model ...

    def forward(self, point_interpreted, dir_interpreted, combine=False):
        # point_interpreted: Tensor(N, 3, board_size, board_size)
        # dir_interpreted: Tensor(N, 4, 11, board_size, board_size)
        point_encoded = point_interpreted
        # Disregard +5 and -5 lines, as those lead to terminated positions ...
        dir_encoded = dir_interpreted[:, :, 1: -1, :, :]
        # point1 = point_encoded.unsqueeze(1)
        if combine:
            dir_encoded = torch.cat([point_encoded.unsqueeze(1).repeat(1, 4, 1, 1, 1), dir_encoded], dim=2)
            # merged_tensor = torch.cat([tensor1, tensor2], dim=1)
        # reshape by stacking the 4 directions into a grouped channel dimension ...
        new_shape = (dir_encoded.shape[0], dir_encoded.shape[1] * dir_encoded.shape[2],
                     dir_encoded.shape[3], dir_encoded.shape[4])
        dir_encoded = dir_encoded.reshape(new_shape)

        return point_encoded, dir_encoded
