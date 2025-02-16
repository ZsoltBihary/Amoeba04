import torch
from Amoeba import Amoeba
from CoreModels import CoreModelTrivial, CoreModelSimple01, CoreModelBihary01, CoreModelBihary02
from Model import Model
from SearchEngine import SearchEngine
# from torchinfo import summary
# from line_profiler_pycharm import profile
# import time

# Collect parameters in a dictionary
args = {
    'board_size': 15,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_MC': 100000,
    'num_child': 50,
    'num_table': 1,
    'num_agent': 400,
    'leaf_buffer_capacity': 6000,
    'eval_batch_size': 200,
    'num_moves': 10,
    'trainer_buffer_capacity': 100000,
    'agent_multi': 32,
    'symmetry_used': True
    # 'split_depth': 0,
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
# core_model = CoreModelBihary02(args, 16, 16, 8, 8, num_blocks=4)
core_model = CoreModelBihary02(args, 32, 32, 16, 16, num_blocks=9)
model = Model(game, core_model)
# Load the state dictionary from the file
# state_dict = torch.load('savedModels/Simple01_02_01.pth')
state_dict = torch.load('savedModels/Bihary02_02_02.pth')
# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.cuda()
engine = SearchEngine(args, model)

player = -torch.ones(args.get('num_table'), dtype=torch.int32)
position = game.get_random_positions(n=args.get('num_table'), n_plus=1, n_minus=0)
table = torch.arange(args.get('num_table'))

# Let us monitor a little bit of gameplay ...
game.print_board(position[0])
for i in range(args.get('num_moves')):
    print(i)

    move_policy, position_value = engine.analyze(player, position)

    print("position value = ", position_value[0])
    # print("move policy:\n", torch.round(100 * move_policy[0, :].view(game.board_size, -1)))
    move = torch.argmax(move_policy, dim=1)
    game.move(position, player, move)
    # position[table, move] = player
    # player *= -1
    terminal_state_value, is_terminal = (
        model.check_EOG((player.view(-1, 1) * position).to(dtype=torch.float32, device=args.get('CUDA_device'))))
    game.print_board(position[0], move[0].item())
    if is_terminal[0].item():
        result_value = terminal_state_value.to(device='cpu') * player
        print('Game over, result = ', result_value[0].item())
        break

a = 42
