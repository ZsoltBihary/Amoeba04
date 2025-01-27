import torch
import torch.nn as nn
import torch.nn.functional as F
from Game import Game
from helper_functions import soft_characteristic
from CustomLayers import Point2DirSum


class Amoeba(Game):
    def __init__(self, config: dict):
        super().__init__(config)
        self.board_size = config.get("board_size", 15)
        self.position_size = self.board_size ** 2
        self.action_size = self.board_size ** 2
        self.win_length = config.get("win_length", 5)
        self.CUDA_device = config.get("CUDA_device", "cpu")

        self.symmetry_index, self.inv_symmetry_index = self.calculate_symmetry()

        self.stones = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device=self.CUDA_device)

        # This helper custom CNN layer is used in self.encode to sum stones over lines in four directions
        # sum_kernel = torch.ones(3, device=self.CUDA_device).diag_embed().unsqueeze(2).repeat(1, 1, self.win_length)
        # self.sum_conv = DirectionalConvolution(3, 3,
        #                                        self.win_length, sum_kernel)

        self.sum_conv = Point2DirSum(3, self.win_length, self.CUDA_device)

        # This helper tensor encodes the type of lines with length win_length. E.g., if win_length = 5, then
        # -5: 5 white stones,
        # -4: 4 white stones + 1 empty,
        # -3: 3 white stones + 2 empty, ...
        # +5: 5 black stones
        self.char_list = torch.arange(-self.win_length, self.win_length + 1,
                                      device=self.CUDA_device, dtype=torch.float32)
        return

    def get_empty_positions(self, n: int):
        return torch.zeros((n, self.position_size), dtype=torch.int32)

    def get_random_positions(self, n: int, n_plus: int, n_minus: int):
        position = self.get_empty_positions(n)
        for i in range(n):
            for _ in range(n_plus):
                valid_indices = torch.nonzero(position[i, :] == 0)
                if valid_indices.numel() > 0:
                    # Randomly select one of these indices
                    random_index = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                    position[i, random_index] = 1
            for _ in range(n_minus):
                valid_indices = torch.nonzero(position[i] == 0)
                if valid_indices.numel() > 0:
                    # Randomly select one of these indices
                    random_index = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                    position[i, random_index] = -1
        return position

    def print_board(self, position: torch.Tensor, last_action=-1):
        pos = position.clone()
        if last_action >= 0:
            pos[last_action] *= 2
        value_to_char = {-2: '(O)', -1: ' O ', 0: ' - ', 1: ' X ', 2: '(X)'}
        board = pos.reshape(self.board_size, self.board_size)
        horizontal = " "
        for _ in range(self.board_size * 3):
            horizontal = horizontal + "-"
        print(horizontal)
        # Iterate through each row of the board
        for row in board:
            row_chars = [value_to_char[value.item()] for value in row]
            # row_string = '  '.join(row_chars)
            row_string = ''.join(row_chars)
            print("|" + row_string + "|")
        print(horizontal)

    def move(self, position, player, action):
        # noinspection PyTypeChecker
        if not torch.all(position[torch.arange(position.shape[0]), action] == 0):
            raise ValueError("Illegal move: action must target an empty space (position == 0).")
        position[torch.arange(position.shape[0]), action] = player
        player *= -1

    def encode(self, state):
        board = state.view(state.shape[0], self.board_size, self.board_size)
        point_encoded = soft_characteristic(board, self.stones).permute(0, 3, 1, 2)
        # point_encoded: Tensor(N, 3, board_size, board_size)
        dir_sum = self.sum_conv(point_encoded)
        dir_sum = dir_sum.permute(0, 2, 3, 1)
        dir_sum = dir_sum.reshape(*dir_sum.shape[:-1], 4, 3)
        diff = dir_sum[..., 1] - dir_sum[..., 2]
        penalty1 = 100.0 * (self.win_length - torch.sum(dir_sum, dim=-1))
        penalty2 = 100.0 * (dir_sum[..., 1] * dir_sum[..., 2])
        x = diff + penalty1 + penalty2
        dir_encoded = soft_characteristic(x, self.char_list)
        dir_encoded = dir_encoded.permute(0, 3, 4, 1, 2)
        # dir_encoded: Tensor(N, 4, 2*win_length+1, board_size, board_size)
        return point_encoded, dir_encoded

    def check_terminal_encoded(self, encoded):
        dir_encoded = encoded[1]
        # point_encoded, dir_encoded = encoded
        plus_signal = torch.amax(dir_encoded[:, :, -1, :, :], dim=(1, 2, 3)) > 0.5
        minus_signal = torch.amax(dir_encoded[:, :, 0, :, :], dim=(1, 2, 3)) > 0.5
        draw_signal = torch.amax(dir_encoded, dim=(1, 2, 3, 4)) < 0.5
        return plus_signal, minus_signal, draw_signal

    def calculate_symmetry(self):
        # set up the symmetry transformations in terms of reindexing state vectors
        # there are 8 transformations of a square board
        # flip45_index = torch.zeros(self.position_size, dtype=torch.long,
        #                            device=self.args.get('CUDA_device'))
        # flip_col_index = torch.zeros(self.position_size, dtype=torch.long,
        #                              device=self.args.get('CUDA_device'))
        # rot90_index = torch.zeros(self.position_size, dtype=torch.long,
        #                           device=self.args.get('CUDA_device'))
        flip45_index = torch.zeros(self.position_size, dtype=torch.long)
        flip_col_index = torch.zeros(self.position_size, dtype=torch.long)
        rot90_index = torch.zeros(self.position_size, dtype=torch.long)

        for i in range(self.position_size):
            row = i // self.board_size
            col = i % self.board_size
            flip45_index[i] = self.board_size * col + row
            flip_col_index[i] = self.board_size * row + (self.board_size - 1 - col)
        for i in range(self.position_size):
            rot90_index[i] = flip45_index[flip_col_index[i]]
        # symmetry_index = torch.zeros((8, self.position_size), dtype=torch.long,
        #                              device=self.args.get('CUDA_device'))
        symmetry_index = torch.zeros((8, self.position_size), dtype=torch.long)

        symmetry_index[0, :] = torch.arange(self.position_size)
        symmetry_index[1, :] = rot90_index
        for i in range(self.position_size):
            symmetry_index[2, i] = rot90_index[symmetry_index[1, i]]
        for i in range(self.position_size):
            symmetry_index[3, i] = rot90_index[symmetry_index[2, i]]
        symmetry_index[4, :] = flip45_index
        for i in range(self.position_size):
            symmetry_index[5, i] = rot90_index[symmetry_index[4, i]]
        for i in range(self.position_size):
            symmetry_index[6, i] = rot90_index[symmetry_index[5, i]]
        for i in range(self.position_size):
            symmetry_index[7, i] = rot90_index[symmetry_index[6, i]]
        # most transformations happen to be equal to their inverse, except two
        inv_symmetry_index = symmetry_index.clone()
        inv_symmetry_index[1, :] = symmetry_index[3, :]
        inv_symmetry_index[3, :] = symmetry_index[1, :]

        return symmetry_index, inv_symmetry_index

    def get_symmetry_states(self, state):
        sym_states = state[..., self.symmetry_index[0, :]]
        for i in range(1, 8):
            sym_states = torch.cat((sym_states, state[..., self.symmetry_index[i, :]]), dim=0)
        return sym_states
