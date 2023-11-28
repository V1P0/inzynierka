import numpy as np
import math
from environment import PaperSoccer


class Node:
    def __init__(self, game: PaperSoccer, args, state, player, prev_player, parent=None, action_taken=None):
        self.game: PaperSoccer = game
        self.args = args
        self.state = state
        self.parent = parent
        self.prev_player = prev_player
        self.player = player
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0

        child_state, player = self.game.get_next_state(self.state, action, 1)
        child_state = self.game.change_perspective(child_state, player=player)

        child = Node(self.game, self.args, child_state, self.player*player, self.player, self, action)
        self.children.append(child)
        return child

    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.prev_player)

        if is_terminal:
            if self.prev_player == -1:
                value = self.game.get_opponent_value(value)
            return value*self.player

        rollout_state = self.state
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state, rollout_player = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, rollout_player)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)
                return value*self.player

    def backpropagate(self, value):
        self.value_sum += -value*self.player
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def search(self, state):
        root = Node(self.game, self.args, state, 1, 1)
        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.prev_player)
            if node.prev_player == -1:
                value = self.game.get_opponent_value(value)

            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
