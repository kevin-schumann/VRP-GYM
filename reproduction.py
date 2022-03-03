"""
Reproduction Script for all results presented
"""
import torch
from gym_vrp.envs.vrp import DefaultVRPEnv, DefaultTSPEnv, DemandVRPEnv
from agents.graph_agent import VRPAgent, VRPDemandAgent
from agents.random_agent import RandomAgent
import csv
from argparse import ArgumentParser
from copy import deepcopy


def reproduce(
    seeds: list,
    num_nodes: int,
    batch_size: int,
    csv_path: str,
    model_path: str,
    num_draw: int,
    env_type: str,
):

    with open(csv_path, "w+", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Seed", "Mean Distance"])
    # Evaluate Default Env
    for seed in seeds:
        env = DemandVRPEnv(
            num_nodes=num_nodes,
            batch_size=batch_size,
            num_draw=num_draw,
            seed=seed,
            video_path=f"random_{seed}_20.mp4",
        )

        # env_r = deepcopy(env)

        # vrp_agent = VRPDemandAgent(depot_dim=2, node_dim=3, seed=seed)
        # vrp_agent.model.load_state_dict(torch.load(model_path))

        random_agent = RandomAgent(seed=seed)
        random_agent.eval()

        # loss_a = vrp_agent.evaluate(env)
        loss_r = random_agent(env)

        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            # writer.writerow(["Attention Agent", seed, loss_a.mean().item()])
            writer.writerow(["Random Agent", seed, loss_r.mean().item()])


if __name__ == "__main__":
    parser = ArgumentParser()

    # hparams
    parser.add_argument("--seeds", type=int, nargs="+", default=[69, 88, 123, 420])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--num_draw", type=int, default=6)
    parser.add_argument("--csv_path", type=str, default="reproduction_results.csv")
    parser.add_argument(
        "--model_path", type=str, default="./check_points/model_epoch__tsp_450.pt"
    )
    parser.add_argument("--env_type", type=str, default="TSP")

    args = parser.parse_args()

    reproduce(**vars(args))
