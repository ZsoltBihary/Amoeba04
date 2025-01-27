import torch
import torch.nn as nn
# import torch.nn.functional as F
# from line_profiler_pycharm import profile
from Amoeba import Game


class Model(nn.Module):
    def __init__(self, game: Game, core_model: nn.Module):
        super().__init__()
        self.game = game
        self.core_model = core_model
        self.device = core_model.device
        # Direct handle to game.check_terminal. Let us see if it is necessary ...
        # self.check_terminal = game.check_terminal

    def forward(self, state):
        encoded = self.game.encode(state)
        logit, state_value = self.core_model(encoded)
        return logit, state_value

    def inference(self, state):
        with (torch.no_grad()):
            encoded = self.game.encode(state)
            logit, state_value = self.core_model(encoded)
            plus_mask, minus_mask, draw_mask = self.game.check_terminal_encoded(encoded)
            terminal_state_value = plus_mask.to(dtype=torch.float32) - minus_mask.to(dtype=torch.float32)
            is_terminal = plus_mask | minus_mask | draw_mask
            term = is_terminal.to(dtype=torch.float32)
            state_value = (1.0-term) * state_value + term * terminal_state_value
        return logit, state_value, is_terminal

    def check_EOG(self, state):
        with (torch.no_grad()):
            encoded = self.game.encode(state)
            # policy, state_value = self.core_model(encoded)
            plus_mask, minus_mask, draw_mask = self.game.check_terminal_encoded(encoded)
            terminal_state_value = plus_mask.to(dtype=torch.float32) - minus_mask.to(dtype=torch.float32)
            is_terminal = plus_mask | minus_mask | draw_mask
            # term = is_terminal.to(dtype=torch.float32)
            # state_value = term * terminal_state_value
        return terminal_state_value, is_terminal
