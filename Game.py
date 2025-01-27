import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import soft_characteristic


# TODO: Copied from base_class_design10.py
#   I have the base class Game here, I will separate these ...
class Game:
    def __init__(self, config: dict):
        """
        Initialize the base game structure.
        Args:
            config (dict): Configuration dictionary with game-specific parameters.
        """
        self.config = config
        self.board_size = config.get("board_size", 15)
        self.position_size = self.board_size ** 2
        self.action_size = self.board_size ** 2
        self.win_length = config.get("win_length", 5)
        self.CUDA_device = config.get("CUDA_device", "cpu")

    # TODO: I will not use this method, it is too cumbersome ... Phase it out ...
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
        state = (player.view(-1, 1) * position).to(dtype=torch.float32)
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
            encoded (varied): Encoded states (N, state_size).
        """
        raise NotImplementedError("Game-specific 'encode' method must be implemented.")

    def check_terminal_encoded(self, encoded):
        """
        Check if the encoded state is terminal. To be implemented by the derived class.
        Args:
            encoded (varied): Encoded states (N, state_size).
        Returns:
            terminal_signal (tuple of torch.Tensors): Boolean flags for win / lose / draw (N, 3).
        """
        raise NotImplementedError("Game-specific 'check_terminal' method must be implemented.")

    def check_terminal(self, state):
        """
        Check if the state is terminal.
        Args:
            state (torch.Tensor): Game states (N, state_size).
        Returns:
            terminal_signal (tuple of torch.Tensors): Boolean flags for win / lose / draw (N, 3).
        """
        encoded = self.encode(state)
        terminal_signal = self.check_terminal_encoded(encoded)
        return terminal_signal
