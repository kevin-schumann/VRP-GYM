"""
Reproduction Script for all results presented
"""
import csv
from argparse import ArgumentParser
from copy import deepcopy

import torch

from agents import IRPAgent, TSPAgent, VRPAgent, RandomAgent
from gym_vrp.envs import IRPEnv, TSPEnv, VRPEnv

env_dict = {"TSP": TSPEnv, "VRP": VRPEnv, "IRP": IRPEnv}
agent_dict = {"TSP": TSPAgent, "VRP": VRPAgent, "IRP": IRPAgent}


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

    for seed in seeds:
        env = env_dict[env_type](
            num_nodes=num_nodes, batch_size=batch_size, num_draw=num_draw, seed=seed,
        )
        env_r = deepcopy(env)

        env.enable_video_capturing(
            video_save_path=f"./videos/video_{env_type}_{num_nodes}_{seed}.mp4"
        )

        agent = agent_dict[env_type](seed=seed)
        agent.model.load_state_dict(torch.load(model_path))

        random_agent = RandomAgent(seed=seed)
        random_agent.eval()

        loss_a = agent.evaluate(env)
        loss_r = random_agent(env_r)

        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)

            for agent_loss, random_loss in zip(loss_a, loss_r):
                writer.writerow([f"{env_type}-Agent", seed, agent_loss.item()])
                writer.writerow(
                    [f"{env_type}-Random-Agent", seed, random_loss.mean().item()]
                )


if __name__ == "__main__":
    parser = ArgumentParser()

    # hparams
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 2468, 2048])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--num_draw", type=int, default=3)
    parser.add_argument("--csv_path", type=str, default="reproduction_results.csv")
    parser.add_argument(
        "--model_path", type=str, default="./check_points/model_epoch__tsp_850.pt"
    )
    parser.add_argument("--env_type", type=str, default="TSP")

    args = parser.parse_args()

    print(vars(args))
    reproduce(**vars(args))
