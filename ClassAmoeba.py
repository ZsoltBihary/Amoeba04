import torch
import torch.nn as nn
import torch.nn.functional as F
from ClassLayers import soft_characteristic, DirectionalConvolution


class Amoeba:
    """Encapsulates the rules of a general Amoeba-type game"""
    def __init__(self, args: dict):
        
        self.args = args
        self.board_size = self.args.get('board_size')
        self.win_length = self.args.get('win_length')
        self.action_size = self.board_size * self.board_size  # this is also the state size
        self.symmetry_index, self.inv_symmetry_index = self.calculate_symmetry()
        self.encoder = self.Encoder(args)

    class Encoder(nn.Module):
        """
        Natural encoding of game states.
        This is the basis for checking for terminal states, and to launch evaluation.
        Implemented as a module.
        """
        def __init__(self, args: dict):
            super(Amoeba.Encoder, self).__init__()
            # DONE: This should go to the class Amoeba ...

            self.board_size = args.get('board_size')
            self.win_length = args.get('win_length')
            self.device = args.get('CUDA_device')

            self.stones = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device=self.device)
            kernel_tensor = torch.ones(3, device=self.device).diag_embed().unsqueeze(2).repeat(1, 1, self.win_length)
            self.dir_conv = DirectionalConvolution(3,
                                                   3,
                                                   self.win_length,
                                                   kernel_tensor)
            self.char_list = torch.arange(-self.win_length, self.win_length + 1,
                                          device=self.device, dtype=torch.float32)

    #     def forward(self, state):
    #         board = state.view(state.shape[0], self.board_size, -1)
    #         point_encoded = soft_characteristic(board, self.stones).permute(0, 3, 1, 2)
    #         # point_encoded: Tensor(N, 3, board_size, board_size)
    #
    #         dir_sum = self.dir_conv(point_encoded.repeat(1, 4, 1, 1))
    #         dir_sum = dir_sum.permute(0, 2, 3, 1)
    #         dir_sum = dir_sum.reshape(*dir_sum.shape[:-1], 4, 3)
    #         diff = dir_sum[..., 1] - dir_sum[..., 2]
    #         pen1 = 100.0 * (5 - torch.sum(dir_sum, dim=-1))
    #         pen2 = 100.0 * (dir_sum[..., 1] * dir_sum[..., 2])
    #         x = diff + pen1 + pen2
    #         dir_encoded = soft_characteristic(x, self.char_list)
    #         dir_encoded = dir_encoded.permute(0, 3, 4, 1, 2)
    #         # dir_encoded: Tensor(N, 4, 11, board_size, board_size)
    #
    #         return point_encoded, dir_encoded
    #
    # def test_terminal_encoded(self, encoded):
    #     """
    #     Test for terminal states using encoded representation.
    #     Args:
    #         encoded (torch.Tensor): The encoded state tensor.
    #     Returns:
    #         terminal_signal (torch.Tensor): A binary tensor indicating terminal states.
    #     """
    #     return (encoded.mean(dim=-1) > 1.0).float()  # Example logic
    #
    #
    #
    #
    #
    #
    # def move(self, idx, position, player, action):
    #     # TODO: Implement this in other parts of the project ...
    #     position[idx, action] = player

    def get_empty_positions(self, n_state: int):
        return torch.zeros((n_state, self.action_size), dtype=torch.int32)

    def get_random_positions(self, n_state: int, n_plus: int, n_minus: int) -> torch.Tensor:

        state = self.get_empty_positions(n_state)
        for i in range(n_state):
            for _ in range(n_plus):
                valid_indices = torch.nonzero(state[i] == 0)
                if valid_indices.numel() > 0:
                    # Randomly select one of these indices
                    random_index = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                    state[i, random_index] = 1
            for _ in range(n_minus):
                valid_indices = torch.nonzero(state[i] == 0)
                if valid_indices.numel() > 0:
                    # Randomly select one of these indices
                    random_index = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                    state[i, random_index] = -1
        return state

    def print_board(self, state: torch.Tensor):
        value_to_char = {-1: 'O', 0: '-', 1: 'X'}
        board = state.reshape(self.board_size, self.board_size)
        horizontal = " "
        for _ in range(self.board_size * 3):
            horizontal = horizontal + "-"
        print(horizontal)
        # Iterate through each row of the board
        for row in board:
            row_chars = [value_to_char[value.item()] for value in row]
            row_string = '  '.join(row_chars)
            print("| " + row_string + " |")
        print(horizontal)

    def calculate_symmetry(self):
        # set up the symmetry transformations in terms of reindexing state vectors
        # there are 8 transformations of a square board
        flip45_index = torch.zeros(self.action_size, dtype=torch.long,
                                   device=self.args.get('CUDA_device'))
        flip_col_index = torch.zeros(self.action_size, dtype=torch.long,
                                     device=self.args.get('CUDA_device'))
        rot90_index = torch.zeros(self.action_size, dtype=torch.long,
                                  device=self.args.get('CUDA_device'))

        for i in range(self.action_size):
            row = i // self.board_size
            col = i % self.board_size
            flip45_index[i] = self.board_size * col + row
            flip_col_index[i] = self.board_size * row + (self.board_size - 1 - col)
        for i in range(self.action_size):
            rot90_index[i] = flip45_index[flip_col_index[i]]

        symmetry_index = torch.zeros((8, self.action_size), dtype=torch.long,
                                     device=self.args.get('CUDA_device'))
        symmetry_index[0, :] = torch.arange(self.action_size)
        symmetry_index[1, :] = rot90_index
        for i in range(self.action_size):
            symmetry_index[2, i] = rot90_index[symmetry_index[1, i]]
        for i in range(self.action_size):
            symmetry_index[3, i] = rot90_index[symmetry_index[2, i]]
        symmetry_index[4, :] = flip45_index
        for i in range(self.action_size):
            symmetry_index[5, i] = rot90_index[symmetry_index[4, i]]
        for i in range(self.action_size):
            symmetry_index[6, i] = rot90_index[symmetry_index[5, i]]
        for i in range(self.action_size):
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

    # def get_empty_position(self):
    #     return torch.zeros(self.action_size, dtype=torch.int32)

    # def get_new_position(self, position, player, action):
    #     new_position = position.clone()
    #     if new_position[action] == 0:  # Assuming 0 means the position is unoccupied
    #         new_position[action] = player
    #     else:
    #         raise ValueError("Invalid action: The position is already occupied.")
    #     return new_position
