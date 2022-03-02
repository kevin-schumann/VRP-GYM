"""
Reproduction Script for all results presented
"""
import torch
from gym_vrp.envs.vrp import DefaultVRPEnv
from agents.graph_agent import VRPAgent
from agents.random_agent import RandomAgent
import csv

seeds = [69, 88, 123, 420]
# Load pretrained agent
vrp_agent = VRPAgent(depot_dim=2, node_dim=2)
vrp_agent.load_state_dict(torch.load("/check_points/*insert last model*"))
vrp_agent.eval()

random_agent = RandomAgent()
# Evaluate Default Env
for seed in seeds:
    env = DefaultVRPEnv(num_nodes=20, batch_size=512, num_draw=9, seed=seed)
    loss_a, _, _ = vrp_agent.step(env)
    loss_r = random_agent(env)
    # ... Log looses for example

