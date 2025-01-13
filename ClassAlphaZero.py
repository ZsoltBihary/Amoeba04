import torch
from ClassAmoeba import Amoeba
# from ClassModel import TerminalCheck01, TrivialModel01, TrivialModel02, SimpleModel01, DeepMindModel01
from ClassSearchEngine import SearchEngine
from ClassEvaluator import Evaluator
from ClassTrainerBuffer import TrainerBuffer
from ClassTrainer import Trainer
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

    def get_data(self, table):
        max_move = self.next_move[table]
        return (max_move,
                self.player[table, : max_move], self.position[table, : max_move],
                self.policy[table, : max_move], self.value[table, : max_move])


class AlphaZero:
    def __init__(self, args: dict, game: Amoeba, evaluator: Evaluator):
        self.args = args
        self.game = game
        self.evaluator = evaluator

        # Set up parameters
        self.num_table = args.get('num_table')
        self.max_move_idx = game.action_size + 1
        self.action_size = game.action_size
        self.trainer_buffer_capacity = args.get('trainer_buffer_capacity')
        # self.CUDA_device = args.get('CUDA_device')
        self.num_moves = args.get('num_moves')  # This is the total number of moves during the simulation

        self.search_engine = SearchEngine(args, game, evaluator)
        self.play_history = PlayHistory(self.num_table, self.action_size+1, self.action_size)
        self.trainer_buffer = TrainerBuffer(self.trainer_buffer_capacity, self.action_size)
        self.trainer = Trainer(evaluator.model, self.trainer_buffer)

        self.next_move_idx = torch.zeros(self.num_table, dtype=torch.long)
        # TODO: Precalculate inverse temperatures ...
        # self.inverse_temp = torch.zeros(self.max_move_idx, dtype=torch.float32)
        self.all_table = torch.arange(self.num_table)
        self.player = torch.zeros(self.num_table, dtype=torch.int32)
        self.position = torch.zeros((self.num_table, self.action_size), dtype=torch.int32)
        # self.policy = torch.zeros((self.num_table, self.action_size), dtype=torch.float32)
        # self.value = torch.zeros(self.num_table, dtype=torch.float32)

    def set_start_positions(self, tables):
        n_table = tables.shape[0]
        if n_table == 0:
            return
        self.play_history.empty(tables)
        # TODO: This is a possible start with one +1 and one -1 stone, next player is +1
        #   Could be different ...
        self.position[tables] = self.game.get_random_positions(n_table, n_plus=2, n_minus=1)
        self.player[tables] = -1
        self.next_move_idx[tables] = 0
        return

    def save_history_to_buffer(self, table_idx, result):

        max_move, player, position, policy, value = self.play_history.get_data(table_idx)
        sum_policy = torch.sum(policy, dim=1)
        state = player.view(-1, 1) * position
        # TODO: We may want to make this more sophisticated ...
        state_value = player * result
        # TODO: This is the place to extend data to symmetric equivalent states ...
        self.trainer_buffer.add_batch(state, policy, state_value)

        return

    def check_EOG(self):
        states = self.player.view(-1, 1) * self.position
        state_values, terminal = self.evaluator.check_EOG(states)
        if not torch.any(terminal):
            return
        terminated_tables = self.all_table[terminal]
        n_EOG = terminated_tables.shape[0]
        print('Games terminated = ', n_EOG)
        terminated_players = self.player[terminal]
        terminated_state_values = state_values[terminal]
        # terminated_positions = self.position[terminal, :]
        result_values = terminated_players * terminated_state_values
        # DONE: This is where we want to save data from play history to self-play buffer
        for i in range(n_EOG):
            self.save_history_to_buffer(terminated_tables[i], result_values[i])
        # if terminal[0]:
        #     print('End Of Game! Result = ', result_values[0])

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
            # Analyze position with search engine ...
            move_policy, position_value = self.search_engine.analyze(self.player, self.position)
            # Save players, positions, policies and values to history ...
            self.play_history.add_data(self.player, self.position, move_policy, position_value)
            # TESTING!!! Let us monitor a little bit of self-play ...
            # self.game.print_board(self.position[0])
            # print("position value = ", position_value[0])

            # Make move on all tables ...
            self.make_move(move_policy)
            # Check EOG. For finished games save history to buffer, and restart ...
            self.check_EOG()
            print(i+1,
                  f"  Buffer size: {len(self.trainer_buffer)}",
                  f"  Data count: {self.trainer_buffer.data_count}")

            if self.trainer_buffer.data_count > self.trainer_buffer_capacity // 5:
                print('Training begins ...')
                self.trainer.improve_model()
                # self.trainer_buffer.reset_counter()

        return

# ********************* OLD ALPHAZERO MAIN LOOP *******************************************
#     def self_play(self):
#         # set up ...
#         av_move = 0.0
#
#         current_player = -torch.ones(self.num_table, dtype=torch.int32, device=self.CPU_device)
#         current_state = self.game.get_random_state(self.num_table, 1, 0)
#
#         i_move = 0
#         while True:
#             i_move += 1
#             # self.game.print_board(current_state[0])
#             self.model.eval()
#             inv_temp = self.inv_temp_schedule[self.move_index]
#             # print("inv_temp = ", inv_temp)
#             action, probability, value = self.analyzer.analyze(current_player, current_state, inv_temp)
#             # DONE: This is where we want to save the analysis result in game_history ...
#             self.save_to_history(current_player, current_state, action, probability, value)
#
#             move = self.analyzer.select_move(action, probability)
#             current_player, current_state = self.analyzer.make_move(move)
#             self.move_index += 1
#
#             # Let us check for End of Game (EOG), and return result ...
#             EOG, result = self.check_EOG(current_player, current_state)
#             # result *= current_player
#
#             for i_table in range(self.num_table):
#                 # if a game ends on a table ...
#                 if EOG[i_table]:
#                     # print("Game ", i_table, " ended. Result = ", result[i_table].item())
#
#                     # DONE: This is where we want to save game_history + result to buffer ...
#                     self.save_to_buffer(i_table, result[i_table])
#                     av_move = 0.98 * av_move + 0.02 * self.move_index[i_table].item()
#
#                     # reset i_table ...
#                     current_player[i_table] = -1
#                     current_state[i_table, :] = self.game.get_random_state(1, 1, 0)
#                     self.move_index[i_table] = 0
#
#             print(i_move, ", av_move: ", round(av_move),
#                   ", buff: ", len(self.buffer), ", new data: ", self.buffer.new_data_count)
#
#             if (len(self.buffer) > (self.buffer.size * 2) // 5 and
#                     self.buffer.new_data_count > self.buffer.size // 5):
#                 self.trainer.improve_model()
#                 # for name, param in self.model.named_parameters():
#                 #     print(f"Parameter name: {name}")
#                 #     print(f"Parameter value: {param}")
#                 if i_move > self.max_moves:
#                     break
#             # if i_move >= self.max_moves:
#             #     break


