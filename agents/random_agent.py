import torch
import torch.nn as nn
import numpy as np


class RandomAgent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, env) -> float:
        state = env.get_state()

        done = False
        acc_loss = torch.zeros(size=(state.shape[0],))

        # play game
        while not done:
            # get prediction for current state
            actions = []
            for i in range(state.shape[0]):
                pos_nodes = np.argwhere(state[i, :, 3] == 0).flatten()
                actions.append(np.random.choice(pos_nodes, 1)[0])

            state, loss, done, _ = env.step(np.array(actions)[:, None])
            acc_loss += torch.tensor(loss, dtype=torch.float)

        return acc_loss  # shape (batch_size)

