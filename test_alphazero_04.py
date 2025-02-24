import torch
from Amoeba import Amoeba
from Model import Model
from CoreModels import CoreModelSimple01, CoreModelBihary01, CoreModelBihary02, CoreModelBihary03
# from SearchEngine import SearchEngine
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
    'num_MC': 8000,
    'num_child': 100,
    'num_table': 50,
    'num_agent': 1000,
    'leaf_buffer_capacity': 6000,
    'eval_batch_size': 800,
    'num_moves': 2000,
    'trainer_buffer_capacity': 80000,
    'agent_multi': 8,
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
# core_model = CoreModelBihary01(args, 64, 32)
# core_model = CoreModelBihary02(args, 32, 32, 16, 16, num_blocks=9)
core_model = CoreModelBihary03(args, cen_main=32, dir_main=24,
                               cen_resi=12, num_blocks=0,
                               ch_val=8, mul_att=3
                               )
model = Model(game, core_model)
# Load the state dictionary from the file
state_dict = torch.load('savedModels/Bihary03_03_09.pth')
# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.cuda()

model.eval()

start = time.time()
alpha = AlphaZero(args, model)
alpha.self_play()

# # OK, let us save the model ...
torch.save(model.state_dict(), 'savedModels/Bihary03_03_10.pth')

elapsed_time = (time.time() - start) / 60.0
print(f"Elapsed time: {elapsed_time:.1f} minutes")
n_generated = args.get('num_table') * args.get('num_moves')
print(f"Positions generated: {n_generated:.0f}")
gen_per_minute = n_generated / elapsed_time
print(f"Generated per minute: {gen_per_minute:.0f}")
