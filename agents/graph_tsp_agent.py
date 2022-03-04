import csv
import logging
import os
import time
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from .graph_decoder import GraphDecoder
from .graph_encoder import GraphEncoder

logging.basicConfig(level=logging.DEBUG)


class TSPModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        emb_dim: int,
        hidden_dim: int,
        num_attention_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = GraphEncoder(
            node_input_dim=node_dim,
            embedding_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        )
        self.decoder = GraphDecoder(
            emb_dim=emb_dim, num_heads=8, v_dim=emb_dim, k_dim=emb_dim
        )

        self.model = lambda x, mask, rollout: self.decoder(
            x, mask, rollout=rollout
        )  # remove encoding and make it do it once

    def forward(self, env, rollout=False) -> Tuple[float, float]:
        done = False
        state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)
        acc_loss = torch.zeros(size=(state.shape[0],), device=self.device)
        acc_log_prob = torch.zeros(size=(state.shape[0],), device=self.device)

        emb = self.encoder(x=state[:, :, :2])

        while not done:
            actions, log_prob = self.decoder(
                node_embs=emb, mask=state[:, :, 3], rollout=rollout
            )

            state, loss, done, _ = env.step(actions.cpu().numpy())

            acc_loss += torch.tensor(loss, dtype=torch.float, device=self.device)
            acc_log_prob += log_prob.squeeze().to(self.device)

            state = torch.tensor(env.get_state(), dtype=torch.float, device=self.device)

        self.decoder.reset()

        return acc_loss, acc_log_prob  # shape (batch_size), shape (batch_size)


class TSPAgent:
    def __init__(
        self,
        node_dim: int = 2,
        emb_dim: int = 128,
        hidden_dim: int = 512,
        num_attention_layers: int = 3,
        num_heads: int = 8,
        lr: float = 1e-4,
        csv_path: str = "loss_log.csv",
        seed=69,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.csv_path = csv_path
        self.model = TSPModel(
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        ).to(self.device)

        self.target_model = TSPModel(
            node_dim=node_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_attention_layers=num_attention_layers,
            num_heads=num_heads,
        ).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        env,
        epochs: int = 100,
        eval_epochs: int = 1,
        check_point_dir: str = "./check_points/",
    ):
        logging.info("Start Training")
        with open(self.csv_path, "w+", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Cost", "Advantage", "Time"])

        start_time = time.time()

        for e in range(epochs):
            self.model.train()

            loss_m, loss_b, log_prob = self.step(env, (False, True))
            advantage = loss_m - loss_b
            loss = (advantage * log_prob).mean()

            # backpropagate
            self.opt.zero_grad()
            loss.backward()

            self.opt.step()

            # update model if better
            self.baseline_update(env, eval_epochs)

            logging.info(
                f"Epoch {e} finished - Loss: {loss}, Advantage: {advantage.mean()} Dist: {loss_m.mean()}"
            )

            # log training data
            with open(self.csv_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        e,
                        loss.item(),
                        loss_m.mean().item(),
                        advantage.mean().item(),
                        time.time() - start_time,
                    ]
                )

            self.save_model(episode=e, check_point_dir=check_point_dir)

    def save_model(self, episode: int, check_point_dir: str) -> None:
        if not os.path.exists(check_point_dir):
            os.makedirs(check_point_dir)

        if episode % 50 == 0 and episode != 0:
            torch.save(
                self.model.state_dict(), check_point_dir + f"model_epoch_{episode}.pt",
            )

    def step(self, env, rollouts: Tuple[bool, bool]):
        env.reset()
        env_baseline = deepcopy(env)

        # Go through graph batch and get loss
        loss, log_prob = self.model(env, rollouts[0])
        with torch.no_grad():
            loss_b, _ = self.target_model(env_baseline, rollouts[0])

        return loss, loss_b, log_prob

    def evaluate(self, env):
        self.model.eval()

        with torch.no_grad():
            loss, _ = self.model(env, rollout=True)

        return loss

    def baseline_update(self, env, batch_steps: int = 3):
        """
        Updates the baseline with the current model iff
        it perform significantly better than the baseline.

        Args:
            env (gym.env): Env to step through
            batch_steps (int, optional): How many games to play.
        """
        logging.info("Update Baseline")
        self.model.eval()
        self.target_model.eval()

        current_model_cost = []
        baseline_model_cost = []
        with torch.no_grad():
            for _ in range(batch_steps):
                loss, loss_b, _ = self.step(env, [True, True])

                current_model_cost.append(loss)
                baseline_model_cost.append(loss_b)

        current_model_cost = torch.cat(current_model_cost)
        baseline_model_cost = torch.cat(baseline_model_cost)
        advantage = (current_model_cost - baseline_model_cost).mean()
        _, p_value = stats.ttest_rel(
            current_model_cost.tolist(), baseline_model_cost.tolist()
        )

        if advantage.item() <= 0 and p_value <= 0.05:
            print("replacing baceline")
            self.target_model.load_state_dict(self.model.state_dict())
