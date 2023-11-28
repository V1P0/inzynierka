import numpy as np
from environment import PaperSoccer
from mcts import MCTS


def main():
    game = PaperSoccer()
    args = {
        'C': 1.41,
        'num_searches': 1_000
    }

    bot = MCTS(game, args)
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

    state.playerY = 10
    state.playerX = 1

    while not is_terminal:
        game.print_board(state)
        value, is_terminal = game.get_value_and_terminated(state, 1)
        if is_terminal:
            break
        if bot_playing:
            state = game.change_perspective(state, player=-1)
            probs = bot.search(state)
            action = np.argmax(probs)
            state, player = game.get_next_state(state, action, 1)
            state = game.change_perspective(state, player=-1)
        else:
            valid_moves = game.get_valid_moves(state)
            print("0 - down, 1 - down right, 2 - right, 3 - up right, 4 - up, 5 - up left, 6 - left, 7 - down left")
            action = int(input("Enter action: "))
            action = action_mapper[action]
            if valid_moves[action] == 0:
                print("Invalid move")
                continue
            state, player = game.get_next_state(state, action, 1)
        if player == -1:
            # state = game.change_perspective(state, player=-1)
            bot_playing = not bot_playing
    game.print_history(state)


if __name__ == '__main__':
    main()
