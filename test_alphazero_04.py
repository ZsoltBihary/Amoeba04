import torch
from Amoeba import Amoeba
from Model import Model
from CoreModels import CoreModelSimple01, CoreModelBihary01
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
    'num_MC': 2000,
    'num_child': 50,
    'num_table': 80,
    'num_agent': 1200,
    'leaf_buffer_capacity': 6000,
    'eval_batch_size': 600,
    'num_moves': 10,
    'trainer_buffer_capacity': 100000,
    # 'split_depth': 0,
    'agent_multi': 5,
    'symmetry_used': True
    # 'res_channels': 32,
    # 'hid_channels': 16,
    # 'num_res': 4,
    # 'policy_hid_channels': 32,
    # 'value_hid_dim': 64
}

game = Amoeba(args)
# core_model = CoreModelTrivial(args)
# core_model = CoreModelSimple01(args)
core_model = CoreModelBihary01(args, 64, 32)

model = Model(game, core_model)
# Load the state dictionary from the file
state_dict = torch.load('savedModels/Bihary01_03_01.pth')
# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.cuda()

model.eval()

start = time.time()
alpha = AlphaZero(args, model)
alpha.self_play()

# # OK, let us save the model ...
# torch.save(model.state_dict(), 'savedModels/Simple01_02_01.pth')
torch.save(model.state_dict(), 'savedModels/Bihary01_03_02.pth')

elapsed_time = (time.time() - start) / 60.0
print(f"Elapsed time: {elapsed_time:.1f} minutes")
n_generated = args.get('num_table') * args.get('num_moves')
print(f"Positions generated: {n_generated:.0f}")
gen_per_minute = n_generated / elapsed_time
print(f"Generated per minute: {gen_per_minute:.0f}")

#
#
# game = Amoeba(args)
# encoder = BiharyEncoder01(args)
# # encoder = SimpleEncoder01(args)
# sim_shape = (8, 4, 4)
# res_hidden_shape = sim_shape
# # res_hidden_shape = (10, 4, 4)
# policy_hidden_shape = (12, 6, 6)
# value_dim = 24
# model = SimModel01(args, sim_shape, res_hidden_shape, policy_hidden_shape, value_dim)
# model = SimpleModel01(args)

# model.cuda()
# evaluator = Evaluator(args, encoder, model)
#
# # Load the state dictionary from the file
# state_dict = torch.load('savedModels/SimModel01_01_03.pth')
# # Load the state dictionary into the model
# model.load_state_dict(state_dict)
# model.cuda()
#
# n_state = args.get('num_table')
# player = -torch.ones(n_state, dtype=torch.int32, device=args.get('CPU_device'))
# state = game.get_random_state(n_state, 1, 0)
#
# analyzer = Analyzer(args, evaluator, player, state)
# alpha = AlphaZero(args, game, analyzer)
# alpha.self_play()
#
# # OK, let us save the model ...
# torch.save(model.state_dict(), 'savedModels/SimModel01_01_04.pth')
# # Stop the timer
# end_time = time.time()
# # Calculate the elapsed time
# elapsed_time = (end_time - start_time) / 60.0
# data_gen = args.get('max_moves') * args.get('num_table')
# data_per_minute = round(data_gen / elapsed_time)
#
# print(f"Elapsed time: {elapsed_time:.2f} minutes")
# print("data generated = ", data_gen)
# print("data_per_minute = ", data_per_minute)
#
