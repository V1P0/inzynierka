from environment import PaperSoccer
from alpha_zero import AlphaZero, ResNet
import torch


def main():
    args = {
        'C': 2,
        'num_searches': 600,
        'num_iterations': 8,
        'num_selfPlay_iterations': 5,
        'num_parallel_games': 10,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    game = PaperSoccer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(game, 9, 640, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()


if __name__ == '__main__':
    main()
