from gym_vrp.envs.vrp import DefaultVRPEnv
from agents.graph_agent import VRPAgent

env = DefaultVRPEnv(num_nodes=10, batch_size=12, num_draw=9, seed=0)

agent = VRPAgent(depot_dim=2, node_dim=2)

agent.train(env)
