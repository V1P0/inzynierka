import numpy as np
from copy import deepcopy


class Board:
    def __init__(self, rows: int, columns: int):
        # rows x columns x 8 (8 directions)
        self.connections = np.array([[[True] * 8] * rows] * columns)
        self.history = []
        self.playerX: int = columns // 2
        self.playerY: int = rows // 2
        self.visited = np.array([[False] * rows] * columns)
        self.visited[self.playerX][self.playerY] = True
        self.width: int = columns
        self.height: int = rows
        self.moves = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        for i in range(1, rows - 1):
            self.connections[0][i][7] = False
            self.connections[0][i][6] = False
            self.connections[0][i][5] = False
            self.connections[0][i][4] = False
            self.connections[0][i][0] = False
            self.visited[0][i] = True

            self.connections[columns - 1][i][4] = False
            self.connections[columns - 1][i][3] = False
            self.connections[columns - 1][i][2] = False
            self.connections[columns - 1][i][1] = False
            self.connections[columns - 1][i][0] = False
            self.visited[columns - 1][i] = True

        for i in range(0, columns // 2 - 1):
            self.connections[i][1][2] = False
            self.connections[i][1][3] = False
            self.connections[i][1][4] = False
            self.connections[i][1][5] = False
            self.connections[i][1][6] = False
            self.visited[i][1] = True

            self.connections[i][rows - 2][0] = False
            self.connections[i][rows - 2][1] = False
            self.connections[i][rows - 2][2] = False
            self.connections[i][rows - 2][6] = False
            self.connections[i][rows - 2][7] = False
            self.visited[i][rows - 2] = True

            self.connections[columns - 1 - i][1][2] = False
            self.connections[columns - 1 - i][1][3] = False
            self.connections[columns - 1 - i][1][4] = False
            self.connections[columns - 1 - i][1][5] = False
            self.connections[columns - 1 - i][1][6] = False
            self.visited[columns - 1 - i][1] = True

            self.connections[columns - 1 - i][rows - 2][0] = False
            self.connections[columns - 1 - i][rows - 2][1] = False
            self.connections[columns - 1 - i][rows - 2][2] = False
            self.connections[columns - 1 - i][rows - 2][6] = False
            self.connections[columns - 1 - i][rows - 2][7] = False
            self.visited[columns - 1 - i][rows - 2] = True

        self.connections[columns // 2 - 1][1][6] = False
        self.connections[columns // 2 - 1][1][5] = False
        self.connections[columns // 2 - 1][1][4] = False
        self.visited[columns // 2 - 1][1] = True

        self.connections[columns // 2 - 1][rows - 2][6] = False
        self.connections[columns // 2 - 1][rows - 2][7] = False
        self.connections[columns // 2 - 1][rows - 2][0] = False
        self.visited[columns // 2 - 1][rows - 2] = True

        self.connections[columns // 2 + 1][1][4] = False
        self.connections[columns // 2 + 1][1][3] = False
        self.connections[columns // 2 + 1][1][2] = False
        self.visited[columns // 2 + 1][1] = True

        self.connections[columns // 2 + 1][rows - 2][0] = False
        self.connections[columns // 2 + 1][rows - 2][1] = False
        self.connections[columns // 2 + 1][rows - 2][2] = False
        self.visited[columns // 2 + 1][rows - 2] = True

        self.visited[columns // 2-1][0] = True
        self.visited[columns // 2][0] = True
        self.visited[columns // 2+1][0] = True
        self.visited[columns // 2-1][rows-1] = True
        self.visited[columns // 2][rows-1] = True
        self.visited[columns // 2+1][rows-1] = True

    def move(self, direction):
        # 0 - down, 1 - down right, 2 - right, 3 - up right, 4 - up, 5 - up left, 6 - left, 7 - down left
        moves = self.moves
        changeX, changeY = moves[direction]
        prevX = self.playerX
        prevY = self.playerY
        self.playerX += changeX
        self.playerY += changeY
        self.connections[prevX][prevY][direction] = False
        self.connections[self.playerX][self.playerY][(direction + 4) % 8] = False
        if self.visited[self.playerX][self.playerY]:
            return True
        else:
            self.visited[self.playerX][self.playerY] = True
            return False


class PaperSoccer:
    def __init__(self):
        self.row_count = 13
        self.column_count = 9
        self.action_size = 8

    def get_initial_state(self) -> Board:
        """
        :return: initial state of the game
        """
        return Board(self.row_count, self.column_count)

    def get_next_state(self, state: Board, action, player) -> (Board, int):
        """
        :param state: state of the game
        :param action: 0 - down, 1 - down right, 2 - right, 3 - up right, 4 - up, 5 - up left, 6 - left, 7 - down left
        :param player: who is making the move
        :return: new state after the move, player who is making the next move
        """
        state = deepcopy(state)
        state.history.append(action)
        repeat_player = state.move(action)
        if repeat_player:
            return state, player
        else:
            return state, self.get_opponent(player)

    def get_valid_moves(self, state: Board) -> np.ndarray:
        """
        :param state: state of the game
        :return: valid moves at the current state
        """
        return deepcopy(state.connections[state.playerX][state.playerY])

    def get_value_and_terminated(self, state, player) -> (int, bool):
        """
        :param state: current state of the game
        :param player: player whose turn it is
        :return: value of the game for the player who is making the move, True if the game is over
        """
        valid_moves = self.get_valid_moves(state)
        if valid_moves.sum() == 0:
            return -player, True
        else:
            if state.playerY == 0:
                return player, True
            elif state.playerY == state.height - 1:
                return -player, True
        return 0, False

    def get_opponent(self, player) -> int:
        """
        :param player: player who is making the move
        :return: opponent of the player
        """
        return -player

    def get_opponent_value(self, value) -> int:
        return -value

    def change_perspective(self, state, player):
        """
        :param state: state of the game
        :param player: player
        :return:
        """
        state = deepcopy(state)
        if player == -1:
            state.connections = np.flip(state.connections, (0, 1))
            state.visited = np.flip(state.visited, (0, 1))
            state.playerX = state.width - 1 - state.playerX
            state.playerY = state.height - 1 - state.playerY
            state.history = [(move + 4) % 8 for move in state.history]
            state.connections = np.roll(state.connections, 4, axis=2)
        return state

    def get_encoded_state(self, state):
        # Player position
        player = np.zeros((state.width, state.height), dtype=np.float32)
        player[state.playerX, state.playerY] = 1

        # Possible moves
        possible_moves = np.zeros((state.width, state.height), dtype=np.float32)
        moves = self.get_valid_moves(state)
        for i, move in enumerate(moves):
            if move:
                dx, dy = state.moves[i]
                new_x, new_y = state.playerX + dx, state.playerY + dy
                possible_moves[new_x, new_y] = 1

        # Ensure the connections array is in the shape [8, width, height]
        connections = np.transpose(state.connections, (2, 0, 1))

        encoded_state = np.stack(
            (player, possible_moves, state.visited, connections[0], connections[1], connections[2], connections[3],
                connections[4], connections[5], connections[6], connections[7]),
        )

        return encoded_state

    def print_history(self, state):
        for x in state.history:
            print(x, end="")
        print()

    def print_board(self, state):
        # 0 - down, 1 - down right, 2 - right, 3 - up right, 4 - up, 5 - up left, 6 - left, 7 - down left
        for i in range(state.height*2-1):
            for j in range(state.width*2-1):
                if i%2 != 0 and j%2 != 0:
                    if not state.connections[j//2][i//2][1] and not state.connections[j//2+1][i//2][7]:
                        print("X", end="")
                    elif not state.connections[j//2][i//2][1]:
                        print("\\", end="")
                    elif not state.connections[j//2+1][i//2][7]:
                        print("/", end="")
                    else:
                        print(" ", end="")
                elif i%2 != 0:
                    if not state.connections[j//2][i//2][0]:
                        print("|", end="")
                    else:
                        print(" ", end="")
                elif j%2 != 0:
                    if not state.connections[j//2][i//2][2]:
                        print("-", end="")
                    else:
                        print(" ", end="")
                else:
                    if j//2 == state.playerX and i//2 == state.playerY:
                        print("O", end="")
                    elif state.visited[j//2][i//2]:
                        print("x", end="")
                    else:
                        print(".", end="")
            print()
        print()


