from environment import PaperSoccer
from mcts import MCTS
from alpha_zero import AlphaZero, ResNet, AlphaMCTS
import numpy as np
import torch


def main():
    args = {
        'C': 2,
        'num_searches': 300,
        'num_iterations': 8,
        'num_selfPlay_iterations': 500,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }
    args2 = {
        'C': 2,
        'num_searches': 1000,
        'num_iterations': 8,
        'num_selfPlay_iterations': 500,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    game = PaperSoccer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 128, device)
    model.load_state_dict(torch.load("model_test.pt", map_location=device))
    model.eval()

    bot = AlphaMCTS(game, args, model)
    bot2 = MCTS(game, args2)
    state = game.get_initial_state()
    is_terminal = False
    bot_playing = True
    action_mapper = {
        2: 0,
        3: 1,
        6: 2,
        9: 3,
        8: 4,
        7: 5,
        4: 6,
        1: 7
    }

    # state.playerY = 10
    # state.playerX = 1

    while not is_terminal:
        game.print_board(state)
        value, is_terminal = game.get_value_and_terminated(state, 1)
        if is_terminal:
            break
        if bot_playing:
            probs = bot.search(state)
            print("alpha", probs)
            action = np.argmax(probs)
            state, player = game.get_next_state(state, action, 1)
        else:
            state = game.change_perspective(state, player=-1)
            probs = bot2.search(state)
            print("mcts", probs)
            action = np.argmax(probs)
            state, player = game.get_next_state(state, action, 1)
            state = game.change_perspective(state, player=-1)
        if player == -1:
            # state = game.change_perspective(state, player=-1)
            bot_playing = not bot_playing
    game.print_history(state)


if __name__ == '__main__':
    main()