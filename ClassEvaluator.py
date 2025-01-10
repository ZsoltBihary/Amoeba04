import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
# from ClassModel import evaluate01
from line_profiler_pycharm import profile
# from helper_functions import unique, duplicate_indices


class Evaluator:
    def __init__(self, args: dict, game, terminal_check, model):
        self.args = args
        self.game = game
        self.terminal_check = terminal_check
        self.model = model
        self.CUDA_device = args.get('CUDA_device')

    @profile
    def evaluate(self, state):
        state_CUDA = state.to(device=self.CUDA_device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            term_indicator_CUDA = self.terminal_check(state_CUDA)
            result_CUDA = self.model(state_CUDA)
        term_indicator = term_indicator_CUDA.to(device='cpu', non_blocking=False)
        logit = result_CUDA[0].to(device='cpu', non_blocking=False)
        state_value = result_CUDA[1].to(device='cpu', non_blocking=False)

        # Interpret result ...
        dir_max = term_indicator[:, 0]
        dir_min = term_indicator[:, 1]
        sum_abs = term_indicator[:, 2]
        plus_mask = (dir_max + 0.1 > self.game.win_length)
        minus_mask = (dir_min - 0.1 < -self.game.win_length)
        draw_mask = (sum_abs + 0.1 > self.game.action_size)
        # state_value = torch.zeros_like(state, dtype=torch.float32)
        state_value[draw_mask] = 0.0
        state_value[plus_mask] = 1.0
        state_value[minus_mask] = -1.0
        terminal_mask = plus_mask | minus_mask | draw_mask

        return logit, state_value, terminal_mask

    def check_EOG(self, state):

        state_CUDA = state.to(device=self.CUDA_device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            term_indicator_CUDA = self.terminal_check(state_CUDA)
            # result_CUDA = self.model(states_CUDA)

        term_indicator = term_indicator_CUDA.to(device='cpu', non_blocking=False)
        # logit = result_CUDA[0].to(device='cpu', non_blocking=False)
        # value = result_CUDA[1].to(device='cpu', non_blocking=False)
        # Interpret result ...
        dir_max = term_indicator[:, 0]
        dir_min = term_indicator[:, 1]
        sum_abs = term_indicator[:, 2]
        plus_mask = (dir_max + 0.1 > self.game.win_length)
        minus_mask = (dir_min - 0.1 < -self.game.win_length)
        draw_mask = (sum_abs + 0.1 > self.game.action_size)
        state_value = torch.zeros_like(state, dtype=torch.float32)
        state_value[draw_mask] = 0.0
        state_value[plus_mask] = 1.0
        state_value[minus_mask] = -1.0
        terminal_mask = plus_mask | minus_mask | draw_mask

        return state_value, terminal_mask
