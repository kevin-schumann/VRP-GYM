from gym_vrp.envs import VRPEnv, IRPEnv, TSPEnv
from agents import TSPAgent, IRPAgent, VRPAgent
import time

seeds = [69, 123, 420]
num_nodes = [20, 30, 40]
batch_size = 256

for seed in seeds:
    for num_node in num_nodes:
        env_tsp = TSPEnv(num_nodes=num_node, batch_size=batch_size, seed=seed)
        env_vrp = VRPEnv(num_nodes=num_node, batch_size=batch_size, seed=seed)
        env_irp = IRPEnv(num_nodes=num_node, batch_size=batch_size, seed=seed)

        agent_tsp = TSPAgent(
            seed=seed, csv_path=f"./train_logs/loss_log_tsp_{num_node}_{seed}.csv",
        )
        agent_tsp.train(
            env_tsp,
            epochs=851,
            check_point_dir=f"./check_points/tsp_{num_node}_{seed}/",
        )

        agent_vrp = VRPAgent(
            seed=seed, csv_path=f"./train_logs/loss_log_vrp_{num_node}_{seed}.csv",
        )
        agent_vrp.train(
            env_vrp,
            epochs=851,
            check_point_dir=f"./check_points/vrp_{num_node}_{seed}/",
        )

        agent_irp = IRPAgent(
            seed=seed, csv_path=f"./train_logs/loss_log_irp_{num_node}_{seed}.csv",
        )
        agent_irp.train(
            env_irp,
            epochs=851,
            check_point_dir=f"./check_points/irp_{num_node}_{seed}/",
        )

