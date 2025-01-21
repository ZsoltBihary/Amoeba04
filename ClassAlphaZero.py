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
    def __init__(self, num_table, max_move, position_size):

        self.num_table = num_table
        self.max_move = max_move
        self.position_size = position_size

        self.next_move = torch.zeros(self.num_table, dtype=torch.long)

        self.all_table = torch.arange(self.num_table)

        self.player = torch.zeros((self.num_table, self.max_move), dtype=torch.int32)
        self.position = torch.zeros((self.num_table, self.max_move, self.position_size), dtype=torch.int32)
        self.policy = torch.zeros((self.num_table, self.max_move, self.position_size), dtype=torch.float32)
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
        self.max_move_idx = game.position_size + 1
        self.position_size = game.position_size
        self.trainer_buffer_capacity = args.get('trainer_buffer_capacity')
        # self.CUDA_device = args.get('CUDA_device')
        self.num_moves = args.get('num_moves')  # This is the total number of moves during the simulation

        self.search_engine = SearchEngine(args, game, evaluator)
        self.play_history = PlayHistory(self.num_table, self.position_size+1, self.position_size)
        self.trainer_buffer = TrainerBuffer(self.trainer_buffer_capacity, self.position_size)
        self.trainer = Trainer(evaluator.model, self.trainer_buffer)

        self.next_move_idx = torch.zeros(self.num_table, dtype=torch.long)
        # TODO: Precalculate how many best moves are considered ...
        self.k_move_select = torch.ones(self.max_move_idx, dtype=torch.int32)
        self.k_move_select[: 10] = 10
        self.all_table = torch.arange(self.num_table)
        self.player = torch.zeros(self.num_table, dtype=torch.int32)
        self.position = torch.zeros((self.num_table, self.position_size), dtype=torch.int32)
        # self.policy = torch.zeros((self.num_table, self.position_size), dtype=torch.float32)
        # self.value = torch.zeros(self.num_table, dtype=torch.float32)

    def set_start_positions(self, tables):
        n_table = tables.shape[0]
        if n_table == 0:
            return
        self.play_history.empty(tables)

        self.position[tables] = self.game.get_random_positions(n_table, n_plus=1, n_minus=0)
        self.player[tables] = -1
        self.next_move_idx[tables] = 0
        return

    def save_history_to_buffer(self, table_idx, result):

        max_move, player, position, policy, value = self.play_history.get_data(table_idx)
        state = player.view(-1, 1) * position
        # Deepmind specification: state_value = player * result. I am using a more sophisticated approach ...
        # Calculate moving average for the values ...
        # Smoothing factor
        alpha = 0.3
        # Initialize the averaged tensor
        av_value = torch.empty_like(value)
        av_value[-1] = value[-1]  # Set the last element
        # Compute iteratively going backwards
        for t in range(len(value) - 2, -1, -1):  # Start from the second-to-last element
            av_value[t] = alpha * value[t] + (1 - alpha) * av_value[t + 1]
        memory = 30.0
        beta = 0.9
        w_result = memory / (memory + len(value) - torch.arange(len(value)))
        w_av_value = (1.0 - w_result) * beta
        est_value = w_result * result + w_av_value * av_value
        state_value = player * est_value
        # DONE: This is the place to extend data to symmetric equivalent states ...
        if self.args.get('symmetry_used'):
            state_sym = self.game.get_symmetry_states(state)
            policy_sym = self.game.get_symmetry_states(policy)
            state_value_sym = state_value.repeat(8)
            self.trainer_buffer.add_batch(state_sym, policy_sym, state_value_sym)
        else:
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
        # Move selection is probabilistic, selecting from top k
        for table_idx in range(self.num_table):
            policy = move_policy[table_idx, :]
            move_idx = self.next_move_idx[table_idx]
            k = self.k_move_select[move_idx].item()
            if k == 1:
                move_action = torch.argmax(policy)
            else:
                best_pol, best_actions = torch.topk(policy, k)
                best_pol = best_pol ** (2.0 / k)
                best_probs = best_pol / torch.sum(best_pol)
                idx = torch.multinomial(best_probs, num_samples=1)
                move_action = best_actions[idx]
            self.position[table_idx, move_action] = self.player[table_idx]

        self.player *= -1
        self.next_move_idx += 1
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

            if self.trainer_buffer.data_count > self.trainer_buffer_capacity // 4:
                print('Training begins ...')
                self.trainer.improve_model()
                for name, param in self.evaluator.model.named_parameters():
                    print(f"Parameter name: {name}")
                    print(f"Parameter value: {param}")

        return
