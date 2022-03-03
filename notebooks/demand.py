from gym_vrp.envs.vrp import DemandVRPEnv
from agents.graph_agent import VRPAgent
import numpy as np

env = DemandVRPEnv(num_nodes=20, batch_size=2, num_draw=1, seed=0)

state = env.reset()

print(state)
print(env.load)
state, _, _, _ = env.step(np.array([1, 0])[:, None])
print(env.load)
print(1 - 0.69634349)
print(1 - 0.87650525)
print(state)
