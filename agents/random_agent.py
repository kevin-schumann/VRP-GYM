import torch
import torch.nn as nn
import numpy as np


class RandomAgent(nn.Module):
    def __init__(self, seed: int = 69):
        super().__init__()
        np.random.seed(seed)

    def forward(self, env) -> float:
        state = env.get_state()

        if isinstance(state, tuple):
            state = state[0]

        done = False
        acc_loss = torch.zeros(size=(state.shape[0],))

        # play game
        while not done:
            if isinstance(state, tuple):
                state = state[0]

            # get prediction for current state
            actions = []
            for i in range(state.shape[0]):
                pos_nodes = np.argwhere(state[i, :, -1] == 0).flatten()
                actions.append(np.random.choice(pos_nodes, 1)[0])

            state, loss, done, _ = env.step(np.array(actions)[:, None])

            acc_loss += torch.tensor(loss, dtype=torch.float)

        return acc_loss  # shape (batch_size)
