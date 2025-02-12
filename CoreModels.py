import torch
import torch.nn as nn
import torch.nn.functional as F
from CustomLayers import (Dir2DirSum, Dir2PointConv2D, Point2DirConv2D, Dir2DirConv2D,
                          DepthwiseSeparableConv2D, DirBatchNorm2D)
from line_profiler_pycharm import profile


class FormatDirectionalInput(nn.Module):
    def __init__(self, combine=False):
        super().__init__()
        self.combine = combine
        # self.device = args.get('CUDA_device')
        # self.board_size = args.get('board_size')
        # self.action_size = self.board_size ** 2
        # self.to(self.device)

    def forward(self, encoded):
        point_encoded, dir_encoded = encoded
        # point_encoded: Tensor(N, 3, board_size, board_size)
        # dir_encoded: Tensor(N, 4, 11, board_size, board_size)
        dir_encoded = dir_encoded[:, :, 1: -1, :, :]
        # dir_encoded: Tensor(N, 4, 9, board_size, board_size)
        if self.combine:
            dir_encoded = torch.cat([point_encoded.unsqueeze(1).repeat(1, 4, 1, 1, 1), dir_encoded], dim=2)
        # reshape by stacking the 4 directions into a grouped channel dimension ...
        new_shape = (dir_encoded.shape[0], dir_encoded.shape[1] * dir_encoded.shape[2],
                     dir_encoded.shape[3], dir_encoded.shape[4])
        dir_input = dir_encoded.reshape(new_shape)

        return dir_input


class InputBihary01(nn.Module):
    def __init__(self, num_point, num_per_dir):
        super().__init__()
        self.format_dir_input = FormatDirectionalInput(combine=True)
        self.scale_point = nn.Conv2d(in_channels=3, out_channels=num_point, kernel_size=1, bias=False)
        self.scale_dir = Dir2DirConv2D(in_channels_per_dir=12, out_channels_per_dir=num_per_dir, kernel_size=1)
        # Batch normalization
        self.point_bn = nn.BatchNorm2d(num_features=num_point)
        self.dir_bn = DirBatchNorm2D(num_per_dir=num_per_dir)
        # Activation
        self.point_activation = nn.ReLU()
        self.dir_activation = nn.ReLU()

    def forward(self, encoded):
        point_input = encoded[0]
        point_input = self.scale_point(point_input)

        dir_input = self.format_dir_input(encoded)
        dir_input = self.scale_dir(dir_input)

        point_input = self.point_bn(point_input)
        dir_input = self.dir_bn(dir_input)

        point_input = self.point_activation(point_input)
        dir_input = self.dir_activation(dir_input)

        return point_input, dir_input


class PointDirBihary01(nn.Module):
    def __init__(self, num_point, num_per_dir, kernel_point, kernel_dir, separable=True):
        super().__init__()

        self.dir_dir = Dir2DirConv2D(in_channels_per_dir=num_per_dir, out_channels_per_dir=num_per_dir,
                                     kernel_size=kernel_dir)

        self.dir_point = Dir2PointConv2D(in_channels_per_dir=num_per_dir, out_channels=num_point,
                                         kernel_size=1)

        if separable:
            self.point_point = DepthwiseSeparableConv2D(in_channels=num_point, out_channels=num_point,
                                                        kernel_size=kernel_point)
        else:
            self.point_point = nn.Conv2d(in_channels=num_point, out_channels=num_point,
                                         kernel_size=kernel_point, padding=kernel_point//2, bias=False)
        # Batch normalization
        self.point_bn = nn.BatchNorm2d(num_features=num_point)
        self.dir_bn = DirBatchNorm2D(num_per_dir=num_per_dir)
        # Activation
        self.point_activation = nn.ReLU()
        self.dir_activation = nn.ReLU()

    def forward(self, point_x, dir_x):

        dir_out = self.dir_dir(dir_x)
        point_out = self.point_point(point_x) + self.dir_point(dir_out)

        point_out = self.point_bn(point_out)
        dir_out = self.dir_bn(dir_out)

        point_out = self.point_activation(point_out)
        dir_out = self.dir_activation(dir_out)

        return point_out, dir_out


class PointBihary01(nn.Module):
    def __init__(self, num_in, num_out, kernel_size, separable=True):
        super().__init__()

        if separable:
            self.conv = DepthwiseSeparableConv2D(in_channels=num_in, out_channels=num_out,
                                                 kernel_size=kernel_size)
        else:
            self.conv = nn.Conv2d(in_channels=num_in, out_channels=num_out,
                                  kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        # Batch normalization
        self.bn = nn.BatchNorm2d(num_features=num_out)
        # Activation
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class OutputBihary01(nn.Module):
    def __init__(self, num_in, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=num_in, out_channels=3,
                              kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

        # # Batch normalization
        # self.bn = nn.BatchNorm2d(num_features=num_out)
        # # Activation
        # self.value_activation = nn.ReLU()

    def forward(self, x, free_space):
        out = self.conv(x)

        attention_logit = out[:, 0, ...] + free_space

        policy_logit = out[:, 1, ...] + attention_logit + 999.9 * (free_space - 1.0)
        logit = policy_logit.reshape(policy_logit.shape[0], -1)

        space_value = out[:, 2, ...]

        attention_logit = attention_logit.reshape(attention_logit.shape[0], -1)
        attention = torch.softmax(attention_logit, dim=1)

        space_value = space_value.reshape(space_value.shape[0], -1)
        state_value = torch.einsum('ij,ij->i', attention, space_value)
        state_value = torch.tanh(state_value)

        return logit, state_value


class CoreModelBihary01(nn.Module):
    def __init__(self, args: dict, num_point, num_per_dir):
        super().__init__()
        self.args = args
        self.device = args.get('CUDA_device')
        self.board_size = args.get('board_size')
        self.action_size = self.board_size ** 2
        self.win_length = args.get('win_length')
        self.num_point = num_point
        self.num_per_dir = num_per_dir

        self.format_input = InputBihary01(num_point, num_per_dir)

        self.point_dir_layer1 = PointDirBihary01(num_point, num_per_dir,
                                                 kernel_point=3, kernel_dir=3, separable=True)

        self.point_dir_layer2 = PointDirBihary01(num_point, num_per_dir,
                                                 kernel_point=3, kernel_dir=3, separable=True)

        self.point_dir_layer3 = PointDirBihary01(num_point, num_per_dir,
                                                 kernel_point=3, kernel_dir=3, separable=True)

        self.point_dir_layer4 = PointDirBihary01(num_point, num_per_dir,
                                                 kernel_point=3, kernel_dir=3, separable=True)

        self.point_layer1 = PointBihary01(num_in=num_point, num_out=num_point//2, kernel_size=3, separable=True)

        self.output_layer = OutputBihary01(num_in=num_point//2, kernel_size=3)

        self.to(self.device)

    @profile
    def forward(self, encoded):
        # point_encoded, dir_encoded = encoded
        point_input, dir_input = self.format_input(encoded)
        point_x, dir_x = self.point_dir_layer1(point_input, dir_input)
        point_x, dir_x = self.point_dir_layer2(point_x, dir_x)
        point_x, dir_x = self.point_dir_layer3(point_x, dir_x)
        point_x, dir_x = self.point_dir_layer4(point_x, dir_x)
        point_x = self.point_layer1(point_x)

        free_space = encoded[0][:, 0, ...]
        logit, state_value = self.output_layer(point_x, free_space)

        return logit, state_value


class CoreModelSimple01(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.device = args.get('CUDA_device')
        self.board_size = args.get('board_size')
        self.action_size = self.board_size ** 2
        self.win_length = args.get('win_length')

        self.format_dir_input = FormatDirectionalInput(combine=False)

        self.line_type_size = 2 * self.win_length - 1
        self.sum_lines = Dir2DirSum(self.line_type_size, self.win_length, self.device)

        logit_par = [18.17, 6.13, 1.98, 0.45, -0.02, 0.89, 2.49, 8.43, 23.64]
        value_par = [-1.37, -0.11, -0.07, -0.05, 0.0, 0.04, 0.07, 0.27, 3.05]

        weight_kernel = torch.tensor([logit_par, value_par], dtype=torch.float32).unsqueeze(2).repeat(1, 1, 5)

        self.weight_conv = Dir2PointConv2D(in_channels_per_dir=self.line_type_size,
                                           out_channels=2,
                                           kernel_size=self.win_length,
                                           para_kernel_init=weight_kernel,
                                           diag_kernel_init=weight_kernel)

        self.to(self.device)

    @profile
    def forward(self, encoded):
        point_encoded, dir_encoded = encoded
        dir_input = self.format_dir_input(encoded)
        sum_input = self.sum_lines(dir_input)
        output = self.weight_conv(sum_input)
        # So far so good ...
        logit = output[:, 0, ...] + 999.0 * (point_encoded[:, 0, ...] - 1.0)
        logit = logit.reshape(logit.shape[0], -1)
        attention = torch.softmax(logit, dim=1)
        value = output[:, 1, ...]
        value = value.reshape(logit.shape[0], -1)
        state_value = torch.einsum('ij,ij->i', attention, value)
        state_value = torch.tanh(state_value)

        return logit, state_value


class CoreModelTrivial(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.device = args.get('CUDA_device')
        self.board_size = args.get('board_size')
        self.action_size = self.board_size ** 2
        self.to(self.device)

    @profile
    def forward(self, encoded):
        point_encoded, dir_encoded = encoded
        logit = 99.9 * (point_encoded[:, 0, :, :].view(-1, self.action_size) - 1.0)
        value = torch.sum(dir_encoded, dim=(1, 2, 3, 4)) * 0.0
        return logit, value

