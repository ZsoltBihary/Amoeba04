import torch
import torch.nn as nn
import torch.nn.functional as F
from ClassLayers import soft_characteristic, DirectionalConvolution


class Game:
    def __init__(self, config: dict):
        """
        Initialize the base game structure.
        Args:
            config (dict): Configuration dictionary with game-specific parameters.
        """
        self.config = config
        # self.board_size = config.get("board_size", 15)
        # self.win_length = config.get("win_length", 5)
        # self.CUDA_device = config.get("CUDA_device", "cpu")

    def calc_state(self, player, position):
        """
        Calculate the game states. It is defined as state = player * position.
        Needed for model evaluation, as models always assume it is the +1 player's turn to move.
        Args:
            player (torch.Tensor): Players to move (N,).
            position (torch.Tensor): Game positions (N, state_size).
        Returns:
            state (torch.Tensor): Game states (N, state_size).
        """
        state = player.view(-1, 1) * position
        return state

    def move(self, position, player, action):
        """
        Perform a move in the game. To be implemented by the derived class.
        Args:
            position (torch.Tensor): Game positions (N, state_size).
            player (torch.Tensor): Players making the moves (N,).
            action (torch.Tensor): Actions to perform (N,).
        Returns:
            None, position and player are modified in-place.
        """
        raise NotImplementedError("Game-specific 'move' method must be implemented.")

    def encode(self, state):
        """
        Encode the game state. To be implemented by the derived class.
        Args:
            state (torch.Tensor): Game states (N, state_size).
        Returns:
            encoded (torch.Tensor): Encoded states (N, state_size).
        """
        raise NotImplementedError("Game-specific 'encode' method must be implemented.")

    def check_terminal_encoded(self, encoded):
        """
        Check if the encoded state is terminal. To be implemented by the derived class.
        Args:
            encoded (torch.Tensor): Encoded states (N, state_size).
        Returns:
            terminal_signal (torch.Tensor): Boolean flags for win / lose / draw (N, 3).
        """
        raise NotImplementedError("Game-specific 'check_terminal' method must be implemented.")

    def check_terminal(self, state):
        """
        Check if the state is terminal.
        Args:
            state (torch.Tensor): Game states (N, state_size).
        Returns:
            terminal_signal (torch.Tensor): Boolean flags for win / lose / draw (N, 3).
        """
        encoded = self.encode(state)
        terminal_signal = self.check_terminal_encoded(encoded)
        return terminal_signal


class Amoeba(Game):
    def __init__(self, config: dict):
        super().__init__(config)
        self.board_size = config.get("board_size", 15)
        self.position_size = self.board_size ** 2
        # self.action_size =
        self.win_length = config.get("win_length", 5)
        self.CUDA_device = config.get("CUDA_device", "cpu")
        self.stones = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device=self.CUDA_device)
        # This helper custom CNN layer is used in self.encode to sum stones over lines in four directions
        sum_kernel = torch.ones(3, device=self.CUDA_device).diag_embed().unsqueeze(2).repeat(1, 1, self.win_length)
        self.sum_conv = DirectionalConvolution(3, 3,
                                               self.win_length, sum_kernel)
        # This helper tensor encodes the type of lines with length win_length. E.g., if win_length = 5, then
        # -5: 5 white stones,
        # -4: 4 white stones + 1 empty,
        # -3: 3 white stones + 2 empty, ...
        # +5: 5 black stones
        self.char_list = torch.arange(-self.win_length, self.win_length + 1,
                                      device=self.CUDA_device, dtype=torch.float32)

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

    def print_board(self, position: torch.Tensor):
        value_to_char = {-1: 'O', 0: '-', 1: 'X'}
        board = position.reshape(self.board_size, self.board_size)
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

    def move(self, position, player, action):
        # if position.dim() != 2 or player.dim() != 1 or action.dim() != 1:
        #     raise ValueError("Position must be 2D, and player and action must be 1D tensors.")
        # if not torch.all((action >= 0) & (action < position.shape[1])):
        #     raise ValueError("Actions must be valid indices for the position tensor.")
        position[torch.arange(position.shape[0]), action] = player
        player *= -1

    def encode(self, state):
        board = state.view(state.shape[0], self.board_size, self.board_size)
        point_encoded = soft_characteristic(board, self.stones).permute(0, 3, 1, 2)
        # point_encoded: Tensor(N, 3, board_size, board_size)
        dir_sum = self.sum_conv(point_encoded.repeat(1, 4, 1, 1))
        dir_sum = dir_sum.permute(0, 2, 3, 1)
        dir_sum = dir_sum.reshape(*dir_sum.shape[:-1], 4, 3)
        diff = dir_sum[..., 1] - dir_sum[..., 2]
        penalty1 = 100.0 * (5 - torch.sum(dir_sum, dim=-1))
        penalty2 = 100.0 * (dir_sum[..., 1] * dir_sum[..., 2])
        x = diff + penalty1 + penalty2
        dir_encoded = soft_characteristic(x, self.char_list)
        dir_encoded = dir_encoded.permute(0, 3, 4, 1, 2)
        # dir_encoded: Tensor(N, 4, 11, board_size, board_size)
        return point_encoded, dir_encoded

    def check_terminal_encoded(self, encoded):
        dir_encoded = encoded[1]
        # point_encoded, dir_encoded = encoded
        plus_signal = torch.amax(dir_encoded[:, :, -1, :, :], dim=(1, 2, 3)) > 0.5
        minus_signal = torch.amax(dir_encoded[:, :, 0, :, :], dim=(1, 2, 3)) > 0.5
        draw_signal = torch.amax(dir_encoded, dim=(1, 2, 3, 4)) < 0.5
        return plus_signal, minus_signal, draw_signal


class Model(nn.Module):
    def __init__(self, game: Game, core_model: nn.Module):
        super().__init__()
        self.game = game
        self.core_model = core_model

    def forward(self, state):
        encoded = self.game.encode(state)
        result = self.core_model(encoded)
        return result

    def inference(self, state):
        encoded = self.game.encode(state)
        result = self.core_model(encoded)
        terminal_signal = self.game.check_terminal_encoded(encoded)
        return result, terminal_signal
