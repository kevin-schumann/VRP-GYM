from gym_vrp.envs.vrp import DefaultVRPEnv, DemandVRPEnv
from agents.graph_agent import VRPAgent, VRPDemandAgent

env = DemandVRPEnv(num_nodes=20, batch_size=512, num_draw=9, seed=0)

# agent = VRPAgent(depot_dim=2, node_dim=2)
agent = VRPDemandAgent(depot_dim=2, node_dim=3)
agent.train(env)
