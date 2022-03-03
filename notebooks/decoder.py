from agents.graph_encoder import GraphDemandEncoder
from gym_vrp.envs.vrp import DemandVRPEnv, DefaultVRPEnv
from agents.graph_agent import VRPAgent
import torch

env = DemandVRPEnv(num_nodes=4, batch_size=2, num_draw=1, seed=0)

encoder = GraphDemandEncoder(2, 3, embedding_dim=4, num_heads=2)

state = torch.tensor(env.reset(), dtype=torch.float)

print(state)

encoder(state[:, :, :3], state[:, :, 3].bool())
