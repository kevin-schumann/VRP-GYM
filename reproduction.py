"""
Reproduction Script for all results presented
"""
import torch
from gym_vrp.envs.vrp import DefaultVRPEnv
from agents.graph_agent import VRPAgent
from agents.random_agent import RandomAgent
import csv
from argparse import ArgumentParser


def reproduce(seeds: list, num_nodes: int, batch_size: int, csv_path: str):

    with open(csv_path, "w+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Seed", "Mean Distance"])
    # Evaluate Default Env
    for seed in seeds:
        env = DefaultVRPEnv(
            num_nodes=num_nodes, batch_size=batch_size, num_draw=9, seed=seed
        )
        vrp_agent = VRPAgent(depot_dim=2, node_dim=2, seed=seed)
        vrp_agent.model.load_state_dict(torch.load("/check_points/*insert last model*"))
        vrp_agent.model.eval()

        random_agent = RandomAgent(seed=seed)
        random_agent.eval()

        loss_a, _, _ = vrp_agent.step(env)
        loss_r = random_agent(env)

        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow("Attention Agent", seed, loss_a.mean())
            writer.writerow("Random Agent", seed, loss_r.mean())


if __name__ == "__main__":
    parser = ArgumentParser()

    # hparams
    parser.add_argument("--seeds", type=int, nargs="+", default=[69, 88, 123, 420])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_nodes", type=int, default=50)
    parser.add_argument("--csv_path", type=str, default="reproduction_results.csv")

    args = parser.parse_args()

    reproduce(**vars(args))
