import torch
import numpy as np
from ClassAmoeba import Amoeba
from ClassCoreModel import TerminalCheck01, DeepMindModel01, SimpleModel01
# from ClassEvaluator import EvaluationBuffer
from torchinfo import summary
from line_profiler_pycharm import profile
import time
# from ClassesCNN import soft_characteristic, one_hot_encode, directional_sum, AmoebaTerminal
from ClassLayers import AmoebaInterpreter, AmoebaEncoder, AmoebaTerminal

# Collect parameters in a dictionary
args = {
    'board_size': 7,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    'num_child': 40,
    'num_table': 2,
    'num_MC': 100,
    'num_moves': 5,
    'eval_batch_size': 128,
    'res_channels': 32,
    'hid_channels': 16,
    'num_res': 4,
    'policy_hid_channels': 32,
    'value_hid_dim': 64
}

game = Amoeba(args)
interpreter = AmoebaInterpreter(args)
encoder = AmoebaEncoder(args)
terminal_check = AmoebaTerminal(args)
positions = game.get_random_positions(2, 20, 24).to(dtype=torch.float32)
position_CUDA = positions.cuda()
game.print_board(positions[0, :])

point_interpreted, dir_interpreted = interpreter(position_CUDA)
print("\ninterpreted Shapes:")
print(point_interpreted.shape, " , ", dir_interpreted.shape)

plus_signal, minus_signal, draw_signal = terminal_check(position_CUDA)
print("\nplus_signal, minus_signal, draw_signal:")
print(plus_signal, " , ", minus_signal, " , ", draw_signal)

point_encoded, dir_encoded = encoder(point_interpreted, dir_interpreted, combine=True)
print("\nencoded Shapes:")
print(point_encoded.shape, " , ", dir_encoded.shape)

# game.print_board(positions[0, :])
# print(torch.round(x[0]).to(dtype=torch.int32))

# terminal_check = TerminalCheck01(args)
# # model = DeepMindModel01(args)
# model = SimpleModel01(args)
# model.eval()

# logit, value = model(position_CUDA)
# logit_int = (1 * logit[0].reshape(-1, args.get('board_size'))).to(CUDA_device='cpu').detach().to(dtype=torch.int32).numpy()
# print(logit_int)
# print(value[0])
#
# summary(model, input_data=position_CUDA, verbose=1)
