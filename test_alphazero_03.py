import torch
from Amoeba import Amoeba
from Model import Model
from CoreModels import CoreModelSimple01
from SearchEngine import SearchEngine
# from ClassEvaluator import Evaluator
from AlphaZero import AlphaZero
# from torchinfo import summary
# from line_profiler_pycharm import profile
import time

# Collect parameters in a dictionary
args = {
    'board_size': 15,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_MC': 200,
    'num_child': 40,
    'num_table': 50,
    'num_agent': 150,
    'leaf_buffer_capacity': 4000,
    'eval_batch_size': 200,
    'num_moves': 50,
    'trainer_buffer_capacity': 10000,
    'symmetry_used': True,
    # 'res_channels': 32,
    # 'hid_channels': 16,
    # 'num_res': 4,
    # 'policy_hid_channels': 32,
    # 'value_hid_dim': 64
}

game = Amoeba(args)
# core_model = CoreModelTrivial(args)
core_model = CoreModelSimple01(args)
model = Model(game, core_model)
engine = SearchEngine(args, model)

# game = Amoeba(args)
# # terminal_check = TerminalCheck01(args)
# # model = TrivialModel01(args)
# # model = TrivialModel02(args)
# model = SimpleModel01(args)
# # model = DeepMindModel01(args)
# model.eval()
# evaluator = Evaluator(args, game, terminal_check, model)

start = time.time()
alpha = AlphaZero(args, model)
alpha.self_play()

elapsed_time = (time.time() - start) / 60.0
print(f"Elapsed time: {elapsed_time:.1f} minutes")
n_generated = args.get('num_table') * args.get('num_moves')
print(f"Positions generated: {n_generated:.0f}")
gen_per_minute = n_generated / elapsed_time
print(f"Generated per minute: {gen_per_minute:.0f}")
