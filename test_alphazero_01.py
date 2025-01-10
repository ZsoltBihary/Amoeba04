import torch
from ClassAmoeba import Amoeba
from ClassModel import TerminalCheck01, TrivialModel01, TrivialModel02, SimpleModel01, DeepMindModel01
# from ClassSearchEngine import SearchEngine
from ClassEvaluator import Evaluator
from ClassAlphaZero import AlphaZero
# from torchinfo import summary
# from line_profiler_pycharm import profile
# import time

# Collect parameters in a dictionary
args = {
    'board_size': 15,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_MC': 500,
    'num_child': 40,
    'num_table': 200,
    'num_agent': 700,
    'num_moves': 200,
    'leaf_buffer_size': 2000,
    'eval_batch_size': 800,
    'res_channels': 32,
    'hid_channels': 16,
    'num_res': 4,
    'policy_hid_channels': 32,
    'value_hid_dim': 64
}

game = Amoeba(args)
terminal_check = TerminalCheck01(args)
# model = TrivialModel01(args)
# model = TrivialModel02(args)
model = SimpleModel01(args)
# model = DeepMindModel01(args)
model.eval()
evaluator = Evaluator(args, game, terminal_check, model)
# engine = SearchEngine(args, game, evaluator)

alpha = AlphaZero(args, game, evaluator)
alpha.self_play()
