import torch
import torch.nn as nn
import torch.nn.functional as F
# from ClassLayers import DirectionalConvolution
from CustomLayers import Dir2DirSum, Dir2PointConv2D
from line_profiler_pycharm import profile


class FormatDirectionalInput(nn.Module):
    def __init__(self, args: dict, combine=False):
        super().__init__()
        self.combine = combine
        self.device = args.get('CUDA_device')
        self.board_size = args.get('board_size')
        self.action_size = self.board_size ** 2
        self.to(self.device)

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


class CoreModelSimple01(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.device = args.get('CUDA_device')
        self.board_size = args.get('board_size')
        self.action_size = self.board_size ** 2
        self.win_length = args.get('win_length')

        self.format_dir_input = FormatDirectionalInput(args, combine=False)

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
        # DirectionalConvolution(self.line_type_size, 2,
        #                                           1, weight)

        # policy_logit_tensor = torch.tensor(policy_logit_par, dtype=torch.float32)
        # value_plus_tensor = torch.tensor(value_plus_par, dtype=torch.float32)
        # value_minus_tensor = torch.tensor(value_minus_par, dtype=torch.float32)
        # self.policy_logit_w = nn.Parameter(policy_logit_tensor)
        # self.value_plus_w = nn.Parameter(value_plus_tensor)
        # self.value_minus_w = nn.Parameter(value_minus_tensor)
        self.to(self.device)

    @profile
    def forward(self, encoded):
        point_encoded, dir_encoded = encoded
        dir_input = self.format_dir_input(encoded)
        sum_input = self.sum_lines(dir_input)
        output = self.weight_conv(sum_input)
        # So far so good ...
        logit = output[:, 0, ...] + 100.0 * (point_encoded[:, 0, ...] - 1.0)
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


# *************** THIS IS OLD CODE FOR SimpleModel01 ...  *******************************************
        # encoded_input = self.encoder.encode(board_zero, line_type)
        #
        # # def forward(self, encoded_input):
        # # At this point, this model basically implements the logit and value heads ...
        # # So let us pretend we had some (RESNET) feature processing already,
        # #   resulting in x with the same shape as the encoded_input ...
        #
        # x = encoded_input[:, 1:, ...]
        # # The heads work with flattened x,
        # #   so we convert the last two dimensions from 2d board representation to 1d state representation ...
        # x = x.reshape(x.shape[0], x.shape[1], -1)
        # state_zero = encoded_input[:, 0, ...].reshape(x.shape[0], -1)
        # x_minus = x[:, 0:6, ...]
        # x_plus = x[:, 5:11, ...]
        # # b_size = x.shape[0]
        # policy_logit = torch.einsum('bfi,f->bi', x, self.policy_logit_w) + 10000.0 * state_zero - 10000.0
        # # flat_policy_logit = policy_logit.reshape((b_size, -1))
        #
        # policy = torch.softmax(policy_logit, dim=1) * state_zero
        # policy = F.normalize(policy, p=1, dim=1)
        # # print("sum(logit) = ", torch.sum(logit, dim=1))
        # # logit is an estimation for the action probabilities played by the first player ...
        # # Here we estimate the action probabilities p_j played by the second player ...
        # p_i = torch.clamp(policy, max=0.99)
        # odds_i = p_i / (1.0 - p_i)
        # sum_odds = torch.sum(odds_i, dim=1, keepdim=True)
        # sum_odds_not_i = sum_odds - odds_i
        # p_j = p_i * sum_odds_not_i
        # # print("sum(p_j)", torch.sum(p_j, dim=1))
        # point_value_plus = (torch.einsum('bfi,f->bi', x_plus, self.value_plus_w)) * p_i
        # point_value_minus = (torch.einsum('bfi,f->bi', x_minus, self.value_minus_w)) * p_j
        # value = torch.sum(point_value_plus + point_value_minus, dim=1)
        # value = torch.tanh(value)
        #
        # return policy_logit, value

# ******************************* OLD STUFF *******************************************
# ******************************* OLD STUFF *******************************************
# ******************************* OLD STUFF *******************************************
#
# class SimpleEncoder01:
#     # Interprets board_zero, line_type (dead, -5, ..., +5)
#     # Encodes board_zero + the statistics of line-types (-5, ..., +5) of the 20 lines at each square
#     # Number of features: 1+11 = 12
#     def __init__(self, args: dict):
#         self.args = args
#         self.board_size = self.args.get('board_size')
#         self.device = self.args.get('CUDA_device')
#         self.encoded_input_shape = torch.Size([12, self.board_size, self.board_size])
#         self.encoded_input_dtype = torch.float32
#
#     @profile
#     def interpret(self, state_tensor):
#
#         # CUDA_device = 'cuda'
#         # state_tensor = player.reshape(-1, 1) * state
#         board = state_tensor.reshape(-1, self.board_size, self.board_size).to(torch.long)
#         board_plus = torch.clamp(board, min=0, max=1)
#         board_minus = torch.clamp(board, min=-1, max=0)
#         board_zero = 1 - board_plus + board_minus
#         sum_plus = torch.ones((board.shape[0], 4, self.board_size, self.board_size),
#                               dtype=torch.long, device=self.device)
#         sum_minus = -torch.ones((board.shape[0], 4, self.board_size, self.board_size),
#                                 dtype=torch.long, device=self.device)
#         # horizontal ********************************
#         sum_plus[:, 0, :, 2:-2] = (board_plus[:, :, :-4] + board_plus[:, :, 1:-3] +
#                                    board_plus[:, :, 2:-2] +
#                                    board_plus[:, :, 3:-1] + board_plus[:, :, 4:])
#         sum_minus[:, 0, :, 2:-2] = (board_minus[:, :, :-4] + board_minus[:, :, 1:-3] +
#                                     board_minus[:, :, 2:-2] +
#                                     board_minus[:, :, 3:-1] + board_minus[:, :, 4:])
#         # vertical ********************************
#         sum_plus[:, 1, 2:-2, :] = (board_plus[:, :-4, :] + board_plus[:, 1:-3, :] +
#                                    board_plus[:, 2:-2, :] +
#                                    board_plus[:, 3:-1, :] + board_plus[:, 4:, :])
#         sum_minus[:, 1, 2:-2, :] = (board_minus[:, :-4, :] + board_minus[:, 1:-3, :] +
#                                     board_minus[:, 2:-2, :] +
#                                     board_minus[:, 3:-1, :] + board_minus[:, 4:, :])
#         # diagonal1 ********************************
#         sum_plus[:, 2, 2:-2, 2:-2] = (board_plus[:, :-4, :-4] + board_plus[:, 1:-3, 1:-3] +
#                                       board_plus[:, 2:-2, 2:-2] +
#                                       board_plus[:, 3:-1, 3:-1] + board_plus[:, 4:, 4:])
#         sum_minus[:, 2, 2:-2, 2:-2] = (board_minus[:, :-4, :-4] + board_minus[:, 1:-3, 1:-3] +
#                                        board_minus[:, 2:-2, 2:-2] +
#                                        board_minus[:, 3:-1, 3:-1] + board_minus[:, 4:, 4:])
#         # diagonal2 ********************************
#         sum_plus[:, 3, 2:-2, 2:-2] = (board_plus[:, :-4, 4:] + board_plus[:, 1:-3, 3:-1] +
#                                       board_plus[:, 2:-2, 2:-2] +
#                                       board_plus[:, 3:-1, 1:-3] + board_plus[:, 4:, :-4])
#         sum_minus[:, 3, 2:-2, 2:-2] = (board_minus[:, :-4, 4:] + board_minus[:, 1:-3, 3:-1] +
#                                        board_minus[:, 2:-2, 2:-2] +
#                                        board_minus[:, 3:-1, 1:-3] + board_minus[:, 4:, :-4])
#
#         alive = (sum_plus * sum_minus >= 0).to(torch.long)
#         sum_index = (sum_plus + sum_minus + 6) * alive
#         line_type = F.one_hot(sum_index, num_classes=12)
#         return board_zero, line_type
#
#     @profile
#     def encode(self, board_zero: torch.Tensor, line_type: torch.Tensor) \
#             -> torch.Tensor:
#
#         encoded_input = torch.sum(line_type, dim=1)
#         # horizontal ********************************
#         encoded_input[:, :, :-4, :] += line_type[:, 0, :, 2:-2, :]
#         encoded_input[:, :, 1:-3, :] += line_type[:, 0, :, 2:-2, :]
#         encoded_input[:, :, 3:-1, :] += line_type[:, 0, :, 2:-2, :]
#         encoded_input[:, :, 4:, :] += line_type[:, 0, :, 2:-2, :]
#         # vertical ********************************
#         encoded_input[:, :-4, :, :] += line_type[:, 1, 2:-2, :, :]
#         encoded_input[:, 1:-3, :, :] += line_type[:, 1, 2:-2, :, :]
#         encoded_input[:, 3:-1, :, :] += line_type[:, 1, 2:-2, :, :]
#         encoded_input[:, 4:, :, :] += line_type[:, 1, 2:-2, :, :]
#         # diagonal1 ********************************
#         encoded_input[:, :-4, :-4, :] += line_type[:, 2, 2:-2, 2:-2, :]
#         encoded_input[:, 1:-3, 1:-3, :] += line_type[:, 2, 2:-2, 2:-2, :]
#         encoded_input[:, 3:-1, 3:-1, :] += line_type[:, 2, 2:-2, 2:-2, :]
#         encoded_input[:, 4:, 4:, :] += line_type[:, 2, 2:-2, 2:-2, :]
#         # diagonal2 ********************************
#         encoded_input[:, :-4, 4:, :] += line_type[:, 3, 2:-2, 2:-2, :]
#         encoded_input[:, 1:-3, 3:-1, :] += line_type[:, 3, 2:-2, 2:-2, :]
#         encoded_input[:, 3:-1, 1:-3, :] += line_type[:, 3, 2:-2, 2:-2, :]
#         encoded_input[:, 4:, :-4, :] += line_type[:, 3, 2:-2, 2:-2, :]
#         # insert indicator for empty squares at feature 0 ...
#         encoded_input[:, :, :, 0] = board_zero
#         return encoded_input.permute(0, 3, 1, 2).to(self.encoded_input_dtype)
#
#
# class SimpleModel01(nn.Module):
#     def __init__(self, args: dict):
#         super(SimpleModel01, self).__init__()
#         self.args = args
#         self.device = args.get('CUDA_device')
#         self.encoder = SimpleEncoder01(args)
#         # At this point, I am providing realistic parameters by hand.
#         # ***** Original values *****
#         # policy_logit_par = [15.0, 7.225, 1.609, 0.630, 0.313,
#         #                     -0.041,
#         #                     0.276, 0.848, 2.492, 100.039, 15.0]
#         # value_plus_par = [0.034, 0.152, 0.288, 0.654, 3.848, 10.0]
#         #
#         # value_minus_par = [-10.0, -1.267, -0.668, -0.343, -0.141, -0.019]
#         # ***********************
#         # ***** After 1. training *****
#         # policy_logit_par = [0.0, 18.95, 6.777, 2.333, 0.5559,
#         #                     0.0375,
#         #                     1.0161, 2.8098, 9.0137, 25.622, 0.0]
#         #
#         # value_plus_par = [-0.0101, 0.0287, 0.0761, 0.2616, 3.0438, 0.0]
#         #
#         # value_minus_par = [0.0, -1.4560, -0.1130, -0.0688, -0.0334, 0.0182]
#         # ***********************
#         # ***** After 2. training *****
#         policy_logit_par = [0.0, 18.17, 6.13, 1.979, 0.451,
#                             -0.0172,
#                             0.896, 2.486, 8.435, 23.64, 0.0]
#
#         value_plus_par = [-0.0085, 0.0431, 0.0737, 0.2668, 3.0476, 0.0]
#
#         value_minus_par = [0.0, -1.3664, -0.1116, -0.0749, -0.0706, 0.0166]
#         # ***********************
#         # name: policy_logit_w
#         # tensor([0.0000, 15.6839, 4.8753, 1.4894, 0.3408, 0.0361, 0.7088, 2.0066,
#         #         6.5340, 21.4006, 0.0000], CUDA_device='cuda:0', requires_grad=True)
#         # name: value_plus_w
#         # tensor([-0.0147, 0.0460, 0.1388, 0.3576, 3.1156, 0.0000], CUDA_device='cuda:0',
#         #        requires_grad=True)
#         # name: value_minus_w
#         # tensor([0.0000, -1.2691, -0.2513, -0.1015, -0.0866, 0.0234], CUDA_device='cuda:0',
#         #        requires_grad=True)
#         # ***********************
#
#         policy_logit_tensor = torch.tensor(policy_logit_par, dtype=torch.float32)
#         value_plus_tensor = torch.tensor(value_plus_par, dtype=torch.float32)
#         value_minus_tensor = torch.tensor(value_minus_par, dtype=torch.float32)
#         self.policy_logit_w = nn.Parameter(policy_logit_tensor)
#         self.value_plus_w = nn.Parameter(value_plus_tensor)
#         self.value_minus_w = nn.Parameter(value_minus_tensor)
#         self.to(self.device)
#
#     @profile
#     def forward(self, state_CUDA):
#
#         board_zero, line_type = self.encoder.interpret(state_CUDA)
#
#         encoded_input = self.encoder.encode(board_zero, line_type)
#
#         # def forward(self, encoded_input):
#         # At this point, this model basically implements the logit and value heads ...
#         # So let us pretend we had some (RESNET) feature processing already,
#         #   resulting in x with the same shape as the encoded_input ...
#
#         x = encoded_input[:, 1:, ...]
#         # The heads work with flattened x,
#         #   so we convert the last two dimensions from 2d board representation to 1d state representation ...
#         x = x.reshape(x.shape[0], x.shape[1], -1)
#         state_zero = encoded_input[:, 0, ...].reshape(x.shape[0], -1)
#         x_minus = x[:, 0:6, ...]
#         x_plus = x[:, 5:11, ...]
#         # b_size = x.shape[0]
#         policy_logit = torch.einsum('bfi,f->bi', x, self.policy_logit_w) + 10000.0 * state_zero - 10000.0
#         # flat_policy_logit = policy_logit.reshape((b_size, -1))
#
#         policy = torch.softmax(policy_logit, dim=1) * state_zero
#         policy = F.normalize(policy, p=1, dim=1)
#         # print("sum(logit) = ", torch.sum(logit, dim=1))
#         # logit is an estimation for the action probabilities played by the first player ...
#         # Here we estimate the action probabilities p_j played by the second player ...
#         p_i = torch.clamp(policy, max=0.99)
#         odds_i = p_i / (1.0 - p_i)
#         sum_odds = torch.sum(odds_i, dim=1, keepdim=True)
#         sum_odds_not_i = sum_odds - odds_i
#         p_j = p_i * sum_odds_not_i
#         # print("sum(p_j)", torch.sum(p_j, dim=1))
#         point_value_plus = (torch.einsum('bfi,f->bi', x_plus, self.value_plus_w)) * p_i
#         point_value_minus = (torch.einsum('bfi,f->bi', x_minus, self.value_minus_w)) * p_j
#         value = torch.sum(point_value_plus + point_value_minus, dim=1)
#         value = torch.tanh(value)
#
#         return policy_logit, value
#
#
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
# class TerminalCheck01(nn.Module):
#     # def __init__(self, args: dict, input_sim_shape, output_sim_shape):
#     def __init__(self, args: dict):
#         super(TerminalCheck01, self).__init__()
#         self.board_size = args.get('board_size')
#         self.win_length = args.get('win_length')
#         self.device = args.get('CUDA_device')
#
#         if self.win_length == 5:
#             self.padding = 2
#             kernel = torch.tensor([
#                 [[[0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0],
#                   [1, 1, 1, 1, 1],
#                   [0, 0, 0, 0, 0],
#                   [0, 0, 0, 0, 0]]],
#
#                 [[[0, 0, 1, 0, 0],
#                   [0, 0, 1, 0, 0],
#                   [0, 0, 1, 0, 0],
#                   [0, 0, 1, 0, 0],
#                   [0, 0, 1, 0, 0]]],
#
#                 [[[1, 0, 0, 0, 0],
#                   [0, 1, 0, 0, 0],
#                   [0, 0, 1, 0, 0],
#                   [0, 0, 0, 1, 0],
#                   [0, 0, 0, 0, 1]]],
#
#                 [[[0, 0, 0, 0, 1],
#                   [0, 0, 0, 1, 0],
#                   [0, 0, 1, 0, 0],
#                   [0, 1, 0, 0, 0],
#                   [1, 0, 0, 0, 0]]]
#             ], dtype=torch.float32)
#         else:  # meaning self.win_length == 3
#             self.padding = 1
#             kernel = torch.tensor([
#                 [[[0, 0, 0],
#                   [1, 1, 1],
#                   [0, 0, 0],]],
#
#                 [[[0, 1, 0],
#                   [0, 1, 0],
#                   [0, 1, 0]]],
#
#                 [[[1, 0, 0],
#                   [0, 1, 0],
#                   [0, 0, 1]]],
#
#                 [[[0, 0, 1],
#                   [0, 1, 0],
#                   [1, 0, 0]]]
#             ], dtype=torch.float32)
#
#         self.dir_conv = CustomConvLayer(kernel, self.padding)
#         self.to(self.device)
#
#     @profile
#     def forward(self, state_CUDA):
#         x = state_CUDA.view(state_CUDA.shape[0], 1, self.board_size, self.board_size)
#         dir_sum = self.dir_conv(x)
#         dir_max = torch.amax(dir_sum, dim=(1, 2, 3))
#         dir_min = torch.amin(dir_sum, dim=(1, 2, 3))
#         sum_abs = torch.sum(torch.abs(state_CUDA), dim=1)
#
#         return torch.stack([dir_max, dir_min, sum_abs], dim=1)
#
#
# class TrivialModel02(nn.Module):
#     def __init__(self, args: dict):
#         super(TrivialModel02, self).__init__()
#         self.board_size = args.get('board_size')
#         self.device = args.get('CUDA_device')
#
#         sum_kernel = 0.1 * torch.ones((1, 1, 3, 3), dtype=torch.float32)
#
#         self.sum_conv = CustomConvLayer(sum_kernel, 1)
#         self.to(self.device)
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
#         self.device = args.get('CUDA_device')
#         self.to(self.device)
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
#         self.device = args.get('CUDA_device')
#         # First convolution (input layer)
#         self.input_conv = InputBlock(self.res_channels)
#         # Tower of residual blocks
#         self.res_tower = nn.ModuleList([ResBlock(self.res_channels, self.hid_channels)
#                                         for _ in range(self.num_res)])
#         self.policy_head = PolicyHead01(self.res_channels, self.policy_hid_channels)
#         self.value_head = ValueHead01(self.res_channels,
#                                       self.board_size ** 2,
#                                       self.value_hid_dim)
#         self.to(self.device)
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
