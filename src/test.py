import numpy as np
from environment import PaperSoccer
from mcts import MCTS


def main():
    game = PaperSoccer()
    args = {
        'C': 1.41,
        'num_searches': 1000
    }

    bot = MCTS(game, args)
    state = game.get_initial_state()
    is_terminal = False

    while not is_terminal:
        probs = bot.search(state)
        action = np.argmax(probs)
        state, player = game.get_next_state(state, action, 1)
        if player == -1:
            state = game.change_perspective(state, player)
        is_terminal = game.get_value_and_terminated(state, player)[1]
        game.print_board(state)

    game.print_history(state)


if __name__ == '__main__':
    main()
