import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_convolutions import dir2dir_conv2d, dir2point_conv2d, point2dir_conv2d
from custom_convolutions import directional_pointwise_conv2d, directional_depthwise_conv2d
from line_profiler_pycharm import profile


class BatchNormReLU2D(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Batch normalization
        self.bn = nn.BatchNorm2d(num_features=num_features)
        # Activation
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(x))


class DirectionalPointwiseConv2D(nn.Module):
    def __init__(self, cen_in, cen_out, dir_in, dir_out):
        """
        Custom Pointwise Convolutional Layer for directional feature mixing.

        Args:
            cen_in (int): Number of input central feature channels.
            cen_out (int): Number of output central feature channels.
            dir_in (int): Number of input directional feature channels per direction.
            dir_out (int): Number of output directional feature channels per direction.
        """
        super().__init__()

        # self.cen_in = cen_in
        # self.c_out = cen_out
        # self.d_in = dir_in
        # self.d_out = dir_out

        # Define trainable parameters for each mixing tensor
        self.cen2cen = nn.Parameter(torch.empty(cen_out, cen_in))
        self.par2cen = nn.Parameter(torch.empty(cen_out, dir_in))
        self.dia2cen = nn.Parameter(torch.empty(cen_out, dir_in))
        self.cen2dir = nn.Parameter(torch.empty(dir_out, cen_in))
        self.dir2dir = nn.Parameter(torch.empty(dir_out, dir_in))

        # Initialize the parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameters using Kaiming initialization for stability."""
        nn.init.kaiming_normal_(self.cen2cen, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.par2cen, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dia2cen, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.cen2dir, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dir2dir, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """Applies the custom directional pointwise convolution."""
        return directional_pointwise_conv2d(x, self.cen2cen, self.par2cen, self.dia2cen, self.cen2dir, self.dir2dir)


class DirectionalDepthwiseConv2D(nn.Module):
    def __init__(self, cen_in, dir_in, cen_k, dir_k):
        """
        Custom Depthwise Convolutional Layer with separate handling of central and directional features.

        Args:
            cen_in (int): Number of central feature channels.
            dir_in (int): Number of directional feature channels per direction.
            cen_k (int): Kernel size for central features.
            dir_k (int): Kernel size for directional features.
        """
        super().__init__()

        assert dir_k >= cen_k, "Directional kernel size must not be smaller than central kernel size"

        # self.cen_size = cen_size
        # self.dir_size = dir_size
        # self.cen_k = cen_k
        # self.dir_k = dir_k

        # Define learnable convolution kernels
        self.cen_tensor = nn.Parameter(torch.empty(cen_in, cen_k, cen_k))
        self.dir_tensor = nn.Parameter(torch.empty(dir_in, dir_k))

        # Initialize the parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameters using Kaiming initialization for stability."""
        nn.init.kaiming_normal_(self.cen_tensor, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.dir_tensor, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """Applies the custom directional depthwise convolution."""
        return directional_depthwise_conv2d(x, self.cen_tensor, self.dir_tensor)


class DirectionalSeparableConv2D(nn.Module):
    def __init__(self, cen_in, cen_out, dir_in, dir_out, cen_k, dir_k):
        """
        Custom Depthwise-Separable Convolutional Layer using directional depthwise and pointwise convolutions.

        Args:
            cen_in (int): Number of input central feature channels.
            cen_out (int): Number of output central feature channels.
            dir_in (int): Number of input directional feature channels per direction.
            dir_out (int): Number of output directional feature channels per direction.
            cen_k (int): Kernel size for central features.
            dir_k (int): Kernel size for directional features.
        """
        super().__init__()

        # Depthwise Convolution
        self.depthwise = DirectionalDepthwiseConv2D(cen_in, dir_in, cen_k, dir_k)

        # Pointwise Convolution
        self.pointwise = DirectionalPointwiseConv2D(cen_in, cen_out, dir_in, dir_out)

    @profile
    def forward(self, x):
        """Applies depthwise convolution first, then pointwise convolution."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


if __name__ == "__main__":
    N, H, W = 800, 15, 15
    c_in, c_out, d_in, d_out = 32, 16, 32, 16
    c_k, d_k = 3, 5
    input_tensor = torch.randn(N, c_in + 4 * d_in, H, W)

    print("=== Testing DirectionalDepthwiseConv2D ===")
    d_layer = DirectionalDepthwiseConv2D(c_in, d_in, c_k, d_k)
    d_output = d_layer(input_tensor)
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", d_output.shape)

    print("=== Testing DirectionalPointwiseConv2D ===")
    p_layer = DirectionalPointwiseConv2D(c_in, c_out, d_in, d_out)
    p_output = p_layer(d_output)
    print("Input Shape:", d_output.shape)
    print("Output Shape:", p_output.shape)

    print("=== Testing DirectionalSeparableConv2D ===")
    s_layer = DirectionalSeparableConv2D(c_in, c_out, d_in, d_out, c_k, d_k)
    s_output = s_layer(input_tensor)
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", s_output.shape)


class DirBatchNorm2D(nn.Module):
    def __init__(self, num_per_dir):
        super().__init__()
        self.num_per_dir = num_per_dir

        # Define 4 separate batch normalization layers explicitly
        self.bn1 = nn.BatchNorm2d(num_per_dir)
        self.bn2 = nn.BatchNorm2d(num_per_dir)
        self.bn3 = nn.BatchNorm2d(num_per_dir)
        self.bn4 = nn.BatchNorm2d(num_per_dir)

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == 4 * self.num_per_dir, "Channel count must be 4 * num_per_dir"
        # Split input tensor into 4 groups along the channel dimension
        x1, x2, x3, x4 = torch.chunk(x, chunks=4, dim=1)
        # Apply batch normalization separately
        x1 = self.bn1(x1)
        x2 = self.bn2(x2)
        x3 = self.bn3(x3)
        x4 = self.bn4(x4)
        # Concatenate the outputs back along the channel dimension
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        padding = kernel_size // 2  # Keeps output spatial size the same as input

        # Depthwise Convolution (groups=in_channels ensures separate filters per channel)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels, bias=False)

        # Pointwise Convolution (1x1 convolution for feature mixing)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)  # Depthwise Convolution
        x = self.pointwise(x)  # Pointwise Convolution
        return x


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
            # init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.para_kernel, a=0, mode='fan_in', nonlinearity='leaky_relu')

        if diag_kernel_init is not None:
            self.diag_kernel.data.copy_(diag_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.diag_kernel, a=0, mode='fan_in', nonlinearity='leaky_relu')

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
            with torch.no_grad():
                self.para_kernel.copy_(para_kernel_init)
            # self.para_kernel.data.copy_(para_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.para_kernel, a=0, mode='fan_in', nonlinearity='leaky_relu')

        if diag_kernel_init is not None:
            # TODO: Change into this safer code throughout the reset_parameters() methods in all custom layers !!!
            with torch.no_grad():
                self.diag_kernel.copy_(diag_kernel_init)

            # self.diag_kernel.data.copy_(diag_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.diag_kernel, a=0, mode='fan_in', nonlinearity='leaky_relu')
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
            nn.init.kaiming_uniform_(self.para_kernel, a=0, mode='fan_in', nonlinearity='leaky_relu')

        if diag_kernel_init is not None:
            self.diag_kernel.data.copy_(diag_kernel_init)
        else:
            nn.init.kaiming_uniform_(self.diag_kernel, a=0, mode='fan_in', nonlinearity='leaky_relu')

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
            nn.init.kaiming_uniform_(kernel, a=0, mode='fan_in', nonlinearity='leaky_relu')
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


# if __name__ == "__main__":
#     # Example input tensors
#     dir_inputs = torch.randn(1, 12, 8, 8)  # Batch of 128, 4*3 channels, 8x8 board
#     point_inputs = torch.randn(1, 3, 8, 8)  # Batch of 128, 3 channels, 8x8 board
#
#     # Dir2DirConv2D example
#     dir2dir_layer = Dir2DirConv2D(in_channels_per_dir=3, out_channels_per_dir=6, kernel_size=3)
#     output_dir2dir = dir2dir_layer(dir_inputs)
#     print("Dir2Dir Output Shape:", output_dir2dir.shape)
#
#     # Dir2PointConv2D example
#     dir2point_layer = Dir2PointConv2D(in_channels_per_dir=3, out_channels=8, kernel_size=3)
#     output_dir2point = dir2point_layer(dir_inputs)
#     print("Dir2Point Output Shape:", output_dir2point.shape)
#
#     # Point2DirConv2D example
#     point2dir_layer = Point2DirConv2D(in_channels=3, out_channels_per_dir=1, kernel_size=3)
#     output_point2dir = point2dir_layer(point_inputs)
#     print("Point2Dir Output Shape:", output_point2dir.shape)
#
#     # Point2DirSum example
#     point_inputs = torch.zeros(1, 2, 7, 7)  # Batch of 1,  channel 2, 7x7 board
#     point_inputs[0, 0, 1:3, 3:5] = 1.0
#     point_inputs[0, 1, 1:4, 3:6] = 1.0
#     print("point_inputs:\n", point_inputs)
#     dir_sum_layer = Point2DirSum(in_channels=2, win_length=5)
#     dir_sum = dir_sum_layer(point_inputs)
#     print("Point2DirSum Output Shape:", dir_sum.shape)
#     print("dir_sum:\n", dir_sum)
