import torch
from Model import Model
from SearchBuffer import SearchBufferManager
from SearchTree import SearchTree
from line_profiler_pycharm import profile


class SearchEngine:
    def __init__(self, args: dict, model: Model):
        self.args = args
        self.model = model
        self.game = model.game

        # Set up parameters
        self.num_table = args.get('num_table')
        self.num_child = args.get('num_child')
        self.num_MC = args.get('num_MC')
        self.num_agent = args.get('num_agent')
        self.num_node = (self.num_MC + self.num_agent + 1000) * (self.num_child + 10)
        self.position_size = self.game.position_size
        self.max_depth = self.game.position_size + 1
        self.CUDA_device = args.get('CUDA_device')
        # Set up search buffer manager
        self.buffer_mgr = (
            SearchBufferManager(leaf_capacity=args.get('leaf_buffer_capacity'),
                                child_capacity=args.get('leaf_buffer_capacity') * self.num_child * 2,
                                min_batch_size=args.get('eval_batch_size'), action_size=self.position_size,
                                max_depth=self.max_depth))
        # Set up search tree
        self.tree = SearchTree(num_table=self.num_table,
                               num_node=self.num_node,
                               num_child=self.num_child)
        # Set up root attributes
        self.root_player = torch.zeros(self.num_table, dtype=torch.int32)
        self.root_position = torch.zeros((self.num_table, self.position_size), dtype=torch.int32)
        # Set up search agents attributes
        self.all_agents = torch.arange(self.num_agent)
        self.active = torch.zeros(self.num_agent, dtype=torch.bool)
        self.table = torch.zeros(self.num_agent, dtype=torch.long)
        self.node = torch.zeros(self.num_agent, dtype=torch.long)
        self.depth = torch.zeros(self.num_agent, dtype=torch.long)
        self.player = torch.zeros(self.num_agent, dtype=torch.int32)
        self.position = torch.zeros((self.num_agent, self.position_size), dtype=torch.int32)
        self.path = torch.zeros((self.num_agent, self.max_depth), dtype=torch.long)
        # Set up helper attributes
        self.table_order = torch.zeros(self.num_table, dtype=torch.long)
        self.ucb_penalty = 0.1
        self.av_num_agent = 0.0

    def reset(self, root_player, root_position):
        self.buffer_mgr.reset()
        self.tree.reset()
        self.model.eval()
        self.root_player[:] = root_player
        self.root_position[:, :] = root_position
        self.active[:] = False
        self.table_order = torch.arange(self.num_table)
        self.av_num_agent = 1.0
        return

    @profile
    def get_active_indices(self):
        return (self.all_agents[self.active],
                self.table[self.active],
                self.node[self.active])

    @profile
    def save_leaves(self):
        agent, table, node = self.get_active_indices()
        leaf_agent = agent[self.tree.is_leaf[table, node]]
        if leaf_agent.shape[0] > 0:
            self.buffer_mgr.add_leaves(self.table[leaf_agent],
                                       self.node[leaf_agent],
                                       self.player[leaf_agent],
                                       self.position[leaf_agent, :],
                                       self.path[leaf_agent, :],
                                       self.depth[leaf_agent])
            # free up agents that have found leaves
            self.active[leaf_agent] = False
        return

    @profile
    def activate_agents(self):
        passive_agents = self.all_agents[~self.active]
        num_new = min(self.num_table, passive_agents.shape[0])
        new_tables = self.table_order[: num_new]
        new_agents = passive_agents[: num_new]
        self.active[new_agents] = True
        self.table[new_agents] = new_tables
        self.node[new_agents] = 1

        self.player[new_agents] = self.root_player[new_tables]
        self.position[new_agents, :] = self.root_position[new_tables, :]
        self.path[new_agents, 0] = 1
        self.depth[new_agents] = 1
        return

    @profile
    def save_children(self, table, parent_node, child_node, parent_player):
        # Step 1: Replicate table, parent_node and parent_player to 2d shape
        table_expanded = table.unsqueeze(1).repeat(1, self.num_child)
        parent_node_expanded = parent_node.unsqueeze(1).repeat(1, self.num_child)
        parent_player_expanded = parent_player.unsqueeze(1).repeat(1, self.num_child)
        # Step 2: Flatten all tensors
        tables = table_expanded.flatten()
        parents = parent_node_expanded.flatten()
        children = child_node.flatten()
        players = parent_player_expanded.flatten()
        self.buffer_mgr.add_children(tables, parents, children, players)
        return

    # TODO: Think again about splitting agents ... Commented out for now ...
    def split_agents(self, agent, table, child_node, depth_max):

        passive_agent = self.all_agents[~self.active]
        old_idx = torch.arange(agent.shape[0])[self.depth[agent] <= depth_max]
        old_agent = agent[old_idx]
        old_table = table[old_idx]
        old_child_node = child_node[old_idx, :]
        if passive_agent.shape[0] >= old_agent.shape[0]:
            new_agent = passive_agent[: old_agent.shape[0]]
            self.active[new_agent] = True
            self.table[new_agent] = old_table

            ucb_tensor = self.tree.ucb[old_table.view(-1, 1), old_child_node]
            best_idx = torch.argmax(ucb_tensor, dim=1)
            new_node2 = old_child_node[torch.arange(best_idx.shape[0]), best_idx]
            # Lower ucb for best child to facilitate branching for consecutive paths ...
            self.tree.ucb[old_table, new_node2] -= self.ucb_penalty

            self.node[new_agent] = new_node2
            # TODO: Make this general, based on the Amoeba class ...
            new_action = self.tree.action[old_table, new_node2]
            self.position[new_agent, :] = self.position[old_agent, :]
            self.position[new_agent, new_action] = self.player[old_agent]
            self.player[new_agent] = -self.player[old_agent]
            self.path[new_agent, :] = self.path[old_agent, :]
            self.path[new_agent, self.depth[old_agent]] = new_node2
            self.depth[new_agent] = self.depth[old_agent] + 1

        return

    @profile
    def update_agents(self):
        if not self.active.any():
            # print('No active agents ...')
            return
        # All agents hold nodes already expanded at this point ...
        agent, table, parent_node = self.get_active_indices()
        n_agent = agent.shape[0]
        # self.av_num_agent = self.av_num_agent * 0.9 + n_agent * 0.1
        child_node = self.tree.get_children(table, parent_node)
        # Save all the child information to the child buffer, we will use this info to update ucb ...
        self.save_children(table, parent_node, child_node, self.player[agent])
        # Find the best child node based on current ucb ...
        ucb_tensor = self.tree.ucb[table.view(-1, 1), child_node]
        best_idx = torch.argmax(ucb_tensor, dim=1)
        new_node = child_node[torch.arange(best_idx.shape[0]), best_idx]
        # Lower ucb for best child to facilitate branching for consecutive paths ...
        self.tree.ucb[table, new_node] -= self.ucb_penalty

        # TODO: Splitting for more than 1 table seems to work now ?!? ...
        if self.num_table == 1:
            # Then split agents, speeding up gameplay ...
            self.split_agents(agent, table, child_node, depth_max=4)
            self.split_agents(agent, table, child_node, depth_max=8)
        # Low depth nodes are split into 3, medium depth nodes are split into 2.
        else:
            self.split_agents(agent, table, child_node, depth_max=2)

        # Update agent attributes ...
        self.node[agent] = new_node
        new_action = self.tree.action[table, new_node]
        # TODO: Make this general, based on the Game(Amoeba) class ...
        # self.game.move(self.position[agent, :], self.player[agent], new_action)
        self.position[agent, new_action] = self.player[agent]
        self.player[agent] *= -1
        self.path[agent, self.depth[agent]] = new_node
        self.depth[agent] += 1
        return

    @profile
    def collect_leaves(self):
        self.table_order[:] = torch.argsort(self.tree.count[:, 1])
        self.buffer_mgr.reset()
        while not self.buffer_mgr.batch_full:
            self.activate_agents()
            self.save_leaves()
            self.update_agents()
        # Post-process leaf buffer ...
        self.buffer_mgr.post_process()
        return

    @profile
    def batch_evaluate(self):
        state = self.buffer_mgr.get_states()
        state_CUDA = state.to(device=self.CUDA_device, dtype=torch.float32, non_blocking=True)
        logit_CUDA, state_value_CUDA, is_terminal_CUDA = self.model.inference(state_CUDA)
        logit = logit_CUDA.to(device='cpu', non_blocking=False)
        logit -= torch.abs(state) * 99.9
        state_value = state_value_CUDA.to(device='cpu', non_blocking=False)
        is_terminal = is_terminal_CUDA.to(device='cpu', non_blocking=False)
        self.buffer_mgr.add_eval_results(logit, state_value, is_terminal)
        return

    @profile
    def expand_tree(self):
        # ***** These are modified within expand_tree
        #       tree.is_terminal
        #       tree.next_node
        #       tree.is_leaf
        #       tree.start_child
        #       tree.action
        #       tree.prior
        table, node, player, logit, is_term = self.buffer_mgr.get_expand_data()
        self.tree.is_terminal[table, node] = is_term
        # Only expand non-terminal leaves ...
        to_expand = ~is_term
        exp_table, exp_node = table[to_expand], node[to_expand]
        exp_player, exp_logit = player[to_expand], logit[to_expand]
        exp_children = self.tree.expand(exp_table, exp_node, exp_logit)
        # Save leaf_children data to the EVAL children buffer ...
        self.save_children(exp_table, exp_node, exp_children, exp_player)
        return

    @profile
    def back_propagate(self):
        # ***** These are modified within propagation
        #       tree.count
        #       tree.value_sum
        #       tree.value
        table, node, value, multi = self.buffer_mgr.get_propagate_data()
        self.tree.back_propagate(table, node, value, multi)
        return

    @profile
    def update_ucb(self):
        # ***** This is modified within ucb update
        #       tree.ucb
        table, parent, child, parent_player = self.buffer_mgr.get_ucb_data()
        self.tree.update_ucb(table, parent, child, parent_player)

        return

    @profile
    def analyze(self, player, position):
        self.reset(player, position)
        while True:
            # Collect and evaluate new leaves ...
            self.collect_leaves()
            self.batch_evaluate()
            # Update search tree ...
            self.expand_tree()
            self.back_propagate()
            self.update_ucb()
            # Monitor some quantities ... For testing ...
            # mean_value = tensor
            mean_MC = self.tree.count[:, 1].to(torch.float).mean()
            min_MC = torch.min(self.tree.count[:, 1])
            max_MC = torch.max(self.tree.count[:, 1])
            # print('minMC = ', min_MC.item(), ', maxMC = ', max_MC.item())
            # print(round(self.av_num_agent))
            if mean_MC > self.num_MC:
                break
        # Formulate output ...
        table = torch.arange(self.num_table)
        root = torch.ones(self.num_table, dtype=torch.long)
        position_value = self.tree.value[table, root]
        root_children = self.tree.get_children(table, root)
        counts = (0.01 * self.tree.count[table.view(-1, 1), root_children]) ** 5.0
        actions = self.tree.action[table.view(-1, 1), root_children]
        probs = counts / torch.sum(counts, dim=1, keepdim=True)
        move_policy = torch.zeros((self.num_table, self.position_size), dtype=torch.float32)
        move_policy[table.view(-1, 1), actions] = probs
        # print(torch.round(100.0 * move_policy[0, :]))
        return move_policy, position_value
