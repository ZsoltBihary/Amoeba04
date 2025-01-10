import torch
from ClassAmoeba import Amoeba
# from ClassModel import TerminalCheck01, TrivialModel01, TrivialModel02, SimpleModel01, DeepMindModel01
from ClassSearchEngine import SearchEngine
from ClassEvaluator import Evaluator
# from torchinfo import summary
from line_profiler_pycharm import profile
# import time


class PlayHistory:
    def __init__(self, num_table, max_move, action_size):

        self.num_table = num_table
        self.max_move = max_move
        self.action_size = action_size

        self.next_move = torch.zeros(self.num_table, dtype=torch.long)

        self.all_table = torch.arange(self.num_table)

        self.player = torch.zeros((self.num_table, self.max_move), dtype=torch.int32)
        self.position = torch.zeros((self.num_table, self.max_move, self.action_size), dtype=torch.int32)
        self.policy = torch.zeros((self.num_table, self.max_move, self.action_size), dtype=torch.float32)
        self.value = torch.zeros((self.num_table, self.max_move), dtype=torch.float32)

    def empty(self, table):
        self.next_move[table] = 0
        return

    def add_data(self, players, positions, policies, values):
        self.player[self.all_table, self.next_move] = players
        self.position[self.all_table, self.next_move, :] = positions
        self.policy[self.all_table, self.next_move, :] = policies
        self.value[self.all_table, self.next_move] = values
        self.next_move[:] += 1
        return


class AlphaZero:
    def __init__(self, args: dict, game: Amoeba, evaluator: Evaluator):
        self.args = args
        self.game = game
        self.evaluator = evaluator

        # Set up parameters
        self.num_table = args.get('num_table')
        self.max_move_idx = game.action_size + 1
        self.action_size = game.action_size
        # self.CUDA_device = args.get('CUDA_device')
        self.num_moves = args.get('num_moves')  # This is the total number of moves during the simulation

        self.search_engine = SearchEngine(args, game, evaluator)
        self.play_history = PlayHistory(self.num_table, self.action_size+1, self.action_size)

        self.next_move_idx = torch.zeros(self.num_table, dtype=torch.long)
        # TODO: Precalculate inverse temperatures ...
        self.inverse_temp = torch.zeros(self.max_move_idx, dtype=torch.float32)
        self.all_table = torch.arange(self.num_table)
        self.player = torch.zeros(self.num_table, dtype=torch.int32)
        self.position = torch.zeros((self.num_table, self.action_size), dtype=torch.int32)
        # self.policy = torch.zeros((self.num_table, self.action_size), dtype=torch.float32)
        # self.value = torch.zeros(self.num_table, dtype=torch.float32)

    def set_start_positions(self, tables):
        n_state = tables.shape[0]
        if n_state == 0:
            return
        # TODO: This is a possible start with one +1 and -1 stone, next player is +1
        #   Could be different ...
        self.position[tables] = self.game.get_random_positions(n_state, n_plus=1, n_minus=1)
        self.player[tables] = 1
        self.next_move_idx[:] = 0
        self.play_history.empty(tables)
        return

    def check_EOG(self):
        states = self.player.view(-1, 1) * self.position
        state_values, terminal = self.evaluator.check_EOG(states)
        if not torch.any(terminal):
            return
        terminated_tables = self.all_table[terminal]
        terminated_players = self.player[terminal]
        terminated_state_values = state_values[terminal]
        # terminated_positions = self.position[terminal, :]
        result_values = terminated_players * terminated_state_values

        if terminal[0]:
            print('End Of Game! Result = ', result_values[0])

        # TODO: This is where we want to save data from play history to self-play buffer

        # start new games on terminated tables ...
        self.set_start_positions(terminated_tables)

        return

    def make_move(self, move_policy):
        # TODO: Currently, this is deterministic.
        #   Make it probabilistic, using inverse temperature ...
        move_action = torch.argmax(move_policy, dim=1)
        self.position[self.all_table, move_action] = self.player
        self.player *= -1
        self.next_move_idx[:] += 1
        return

    @profile
    def self_play(self):
        self.set_start_positions(self.all_table)

        for i in range(self.num_moves):

            # Check EOG. For finished games save history to buffer, and restart ...
            self.check_EOG()
            # Analyze position with search engine ...
            move_policy, position_value = self.search_engine.analyze(self.player, self.position)
            # Save players, positions, policies and values to history ...
            self.play_history.add_data(self.player, self.position, move_policy, position_value)
            # TODO: TESTING!!! Let us monitor a little bit of self-play ...
            print(i)
            self.game.print_board(self.position[0])
            print("position value = ", position_value[0])
            # Make move on all tables ...
            self.make_move(move_policy)

        return
