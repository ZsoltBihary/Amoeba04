import torch
import torch.nn as nn
import torch.nn.functional as F
from line_profiler_pycharm import profile
from ClassAmoeba import Game


class Model(nn.Module):
    def __init__(self, game: Game, core_model: nn.Module):
        super().__init__()
        self.game = game
        self.core_model = core_model
        # Direct handle to game.check_terminal. Let us see if it is necessary ...
        self.check_terminal = game.check_terminal

    def forward(self, state):
        encoded = self.game.encode(state)
        policy, state_value = self.core_model(encoded)
        return policy, state_value

    def inference(self, state):
        encoded = self.game.encode(state)
        policy, state_value = self.core_model(encoded)
        terminal_signal = self.game.check_terminal_encoded(encoded)
        return policy, state_value, terminal_signal


# class CustomConvLayer(nn.Module):
#     def __init__(self, kernel, padding):
#         super(CustomConvLayer, self).__init__()
#         # Register the kernel as a buffer (non-trainable)
#         self.register_buffer('kernel', kernel)
#         self.padding = padding
#
#     def forward(self, x):
#         # Apply the convolution using the kernel
#         # Since this is a predefined kernel, we do not have bias
#         return F.conv2d(x, self.kernel, padding=self.padding)
#
#
# class TrivialModel02(nn.Module):
#     def __init__(self, args: dict):
#         super(TrivialModel02, self).__init__()
#         self.board_size = args.get('board_size')
#         self.CUDA_device = args.get('CUDA_device')
#
#         sum_kernel = 0.1 * torch.ones((1, 1, 3, 3), dtype=torch.float32)
#
#         self.sum_conv = CustomConvLayer(sum_kernel, 1)
#         self.to(self.CUDA_device)
#
#     @profile
#     def forward(self, state_CUDA):
#         x = state_CUDA.view(state_CUDA.shape[0], 1, self.board_size, self.board_size)
#         sum_abs_x = self.sum_conv(torch.abs(x)+0.2)
#         # logit head
#         logit = 0.1 * sum_abs_x.reshape(sum_abs_x.shape[0], -1) - 99.9 * torch.abs(state_CUDA)
#         # value head
#         value = torch.sum(state_CUDA, dim=1) * 0.0
#         return logit, value
#
#
# class TrivialModel01(nn.Module):
#     def __init__(self, args: dict):
#         super(TrivialModel01, self).__init__()
#         self.CUDA_device = args.get('CUDA_device')
#         self.to(self.CUDA_device)
#
#     @profile
#     def forward(self, state_CUDA):
#         # # logit head
#         logit = -99.9 * torch.abs(state_CUDA)
#         # value head
#         # value = torch.sum(state_CUDA, dim=1) * 0.05 + 0.02
#         value = torch.sum(state_CUDA, dim=1) * 0.0
#         return logit, value
#
#
# class InputBlock(nn.Module):
#     def __init__(self, res_channels):
#         super(InputBlock, self).__init__()
#
#         self.conv = nn.Conv2d(
#             in_channels=3, out_channels=res_channels,
#             kernel_size=5, stride=1, padding=2, bias=False
#         )
#         self.batch_norm = nn.BatchNorm2d(res_channels)
#         self.relu = nn.ReLU()
#
#     @profile
#     def forward(self, x):
#
#         out = self.conv(x)
#         out = self.batch_norm(out)  # BatchNorm after first Conv
#         out = self.relu(out)
#         return out
#
#
# class ResBlock(nn.Module):
#     def __init__(self, res_channels, hid_channels):
#         super(ResBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=res_channels, out_channels=hid_channels,
#             kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.conv2 = nn.Conv2d(
#             in_channels=hid_channels, out_channels=res_channels,
#             kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.batch_norm1 = nn.BatchNorm2d(hid_channels)
#         self.batch_norm2 = nn.BatchNorm2d(res_channels)
#         self.relu = nn.ReLU()
#
#     @profile
#     def forward(self, x):
#         identity = x  # Shortcut connection
#         out = self.conv1(x)
#         out = self.batch_norm1(out)  # BatchNorm after first Conv
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.batch_norm2(out)  # BatchNorm after second Conv
#         out += identity  # Residual addition
#         out = self.relu(out)
#         return out
#
#
# class PolicyHead01(nn.Module):
#     def __init__(self, res_channels, hid_channels):
#         super(PolicyHead01, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=res_channels, out_channels=hid_channels,
#             kernel_size=5, stride=1, padding=2, bias=False
#         )
#         self.conv2 = nn.Conv2d(
#             in_channels=hid_channels, out_channels=1,
#             kernel_size=5, stride=1, padding=2, bias=False
#         )
#         self.batch_norm1 = nn.BatchNorm2d(hid_channels)
#         # self.batch_norm2 = nn.BatchNorm2d(res_channels)
#         self.relu = nn.ReLU()
#
#     @profile
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batch_norm1(x)  # BatchNorm after first Conv
#         x = self.relu(x)
#         x = self.conv2(x)
#         logit = x.squeeze(1).view(x.shape[0], -1)
#         # These are the logit probabilities
#         return logit
#
#
# class ValueHead01(nn.Module):
#     def __init__(self, res_channels, state_size, hid_dim):
#         super(ValueHead01, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=res_channels, out_channels=1,
#             kernel_size=1, stride=1, padding=0, bias=False
#         )
#         self.batch_norm1 = nn.BatchNorm2d(1)
#         self.relu = nn.ReLU()
#         self.lin1 = nn.Linear(in_features=state_size, out_features=hid_dim,
#                               bias=True)
#         # self.relu = nn.ReLU()
#         self.lin2 = nn.Linear(in_features=hid_dim, out_features=1,
#                               bias=True)
#         self.tanh = nn.Tanh()
#
#     @profile
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.batch_norm1(x)  # BatchNorm after first Conv
#         x = self.relu(x)
#         # board = state_CUDA.view(state_CUDA.shape[0], self.board_size, self.board_size)
#         x = self.lin1(x.view(x.shape[0], -1))
#         x = self.relu(x)
#         x = self.lin2(x)
#         value = self.tanh(x)
#
#         return torch.squeeze(value, dim=1)
#
#
# class DeepMindModel01(nn.Module):
#     def __init__(self, args: dict):
#         super(DeepMindModel01, self).__init__()
#         self.board_size = args.get('board_size')
#         self.res_channels = args.get('res_channels')
#         self.hid_channels = args.get('hid_channels')
#         self.num_res = args.get('num_res')
#         self.policy_hid_channels = args.get('policy_hid_channels')
#         self.value_hid_dim = args.get('value_hid_dim')
#         self.CUDA_device = args.get('CUDA_device')
#         # First convolution (input layer)
#         self.input_conv = InputBlock(self.res_channels)
#         # Tower of residual blocks
#         self.res_tower = nn.ModuleList([ResBlock(self.res_channels, self.hid_channels)
#                                         for _ in range(self.num_res)])
#         self.policy_head = PolicyHead01(self.res_channels, self.policy_hid_channels)
#         self.value_head = ValueHead01(self.res_channels,
#                                       self.board_size ** 2,
#                                       self.value_hid_dim)
#         self.to(self.CUDA_device)
#
#     @profile
#     def forward(self, state_CUDA):
#         # reshape and one-hot-encode the input
#         board = state_CUDA.view(state_CUDA.shape[0], self.board_size, self.board_size)
#         board_plus = torch.clamp(board, min=0, max=1)
#         board_minus = -torch.clamp(board, min=-1, max=0)
#         board_zero = 1 - board_plus - board_minus
#         x = torch.stack([board_zero, board_plus, board_minus], dim=1)
#         # convolution on the encoded input
#         x = self.input_conv(x)
#         # residual tower
#         for res_block in self.res_tower:
#             x = res_block(x)
#         # logit head
#         logit = self.policy_head(x)
#         # value head
#         value = self.value_head(x)
#
#         return logit, value
