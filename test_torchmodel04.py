import torch
from Amoeba import Amoeba
from Model import Model
from CoreModels import CoreModelSimple01
from CoreModels import CoreModelBihary01
from torchinfo import summary
from line_profiler_pycharm import profile
import time

# Collect parameters in a dictionary
args = {
    'board_size': 15,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    'num_child': 40,
    'num_table': 2,
    'num_MC': 100,
    'num_moves': 5,
    'eval_batch_size': 16,
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

player = torch.ones(args.get('num_table'), dtype=torch.int32)
position = game.get_random_positions(n=args.get('num_table'), n_plus=1, n_minus=0)
state_CUDA = (player.view(-1, 1) * position).to(dtype=torch.float32, device=args.get('CUDA_device'))

# game.print_board(position[0, :])
# logit, state_value = model(state_CUDA)
# logit_int = (1 * logit[0].reshape(-1, args.get('board_size'))).to(device='cpu').detach().to(dtype=torch.int32).numpy()
# print(logit_int)
# print(state_value[0])

summary(model, input_data=state_CUDA, verbose=2)

summary(model, input_data=state_CUDA, verbose=1)
