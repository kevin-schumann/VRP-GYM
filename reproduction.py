"""
Reproduction Script for all results presented
"""
import torch
from gym_vrp.envs.vrp import VRPEnv, TSPEnv, IRPEnv
from agents import TSPAgent, IRPAgent, RandomAgent, VRPAgent
import csv
from argparse import ArgumentParser
from copy import deepcopy


env_dict = {"TSP": TSPEnv, "VRP": VRPEnv, "IRP": IRPEnv}
agent_dict = {"TSP": TSPAgent, "VRP": TSPAgent, "IRP": IRPAgent}


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
            num_nodes=num_nodes,
            batch_size=batch_size,
            num_draw=num_draw,
            seed=seed,
            video_save_path=f"./videos/agent_{env_type}_{seed}.mp4",
        )

        env_r = deepcopy(env)
        env_r.video_save_path = None

        agent = agent_dict[env_type](seed=seed)
        agent.model.load_state_dict(torch.load(model_path))

        random_agent = RandomAgent(seed=seed)
        random_agent.eval()

        loss_a = agent.evaluate(env)
        loss_r = random_agent(env_r)

        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([f"{env_type}-Agent", seed, loss_a.mean().item()])
            writer.writerow([f"{env_type}-Random-Agent", seed, loss_r.mean().item()])


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
