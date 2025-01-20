import torch
from ClassAmoeba import Amoeba
from OldClassModel import TerminalCheck01, TrivialModel01, TrivialModel02, SimpleModel01, DeepMindModel01
# from ClassSearchEngine import SearchEngine
from ClassEvaluator import Evaluator
from ClassAlphaZero import AlphaZero
# from torchinfo import summary
# from line_profiler_pycharm import profile
import time

# Collect parameters in a dictionary
args = {
    'board_size': 15,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_MC': 100,
    'num_child': 40,
    'num_table': 100,
    'num_agent': 300,
    'leaf_buffer_capacity': 4000,
    'eval_batch_size': 700,
    'num_moves': 100,
    'trainer_buffer_capacity': 2000,
    'symmetry_used': False,
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

start = time.time()
alpha = AlphaZero(args, game, evaluator)
alpha.self_play()

elapsed_time = (time.time() - start) / 60.0
print(f"Elapsed time: {elapsed_time:.1f} minutes")
n_generated = args.get('num_table') * args.get('num_moves')
print(f"Positions generated: {n_generated:.0f}")
gen_per_minute = n_generated / elapsed_time
print(f"Generated per minute: {gen_per_minute:.0f}")
