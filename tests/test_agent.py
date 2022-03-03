import pytest
import torch
from gym_vrp.envs.vrp import DefaultVRPEnv
from agents.graph_encoder import GraphEncoder, GraphDemandEncoder
from agents.graph_decoder import GraphDecoder
from agents.random_agent import RandomAgent
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def setup():
    seed = 69
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_encoder():
    env = DefaultVRPEnv(num_nodes=8, batch_size=2, num_draw=1)
    state = torch.from_numpy(env.reset()).float()

    encoder = GraphEncoder(node_input_dim=2)
    emb = encoder(state[:, :, :2])

    assert emb.shape == (2, 8, 128)


def test_decoder():
    num_graphs = 2
    num_nodes = 8

    env = DefaultVRPEnv(num_nodes=num_nodes, batch_size=num_graphs, num_draw=1)

    # extract important state info
    state = torch.from_numpy(env.reset()).float()
    depot_idx = torch.argmax(state[:, :, 2], axis=1)

    encoder = GraphEncoder(node_input_dim=2)
    decoder = GraphDecoder(v_dim=128, k_dim=128)

    # encode
    embs = encoder(state[:, :, :2])  # shape -> (num_graphs, num_nodes, emb_size(128))

    # decode with initial mask
    mask = torch.zeros(size=(num_graphs, num_nodes))
    next_node, _ = decoder(embs, mask)
    print(next_node)
    assert torch.equal(
        next_node, torch.tensor([[5], [4]], dtype=torch.long)
    ) or torch.equal(next_node, torch.tensor([[1], [5]], dtype=torch.long))


def test_random_agent():
    num_graphs = 2
    num_nodes = 8

    env = DefaultVRPEnv(num_nodes=num_nodes, batch_size=num_graphs, num_draw=1,)
    agent = RandomAgent()
    loss = agent(env)

    assert np.isclose([loss.mean().item()], [4.857976913452148])

