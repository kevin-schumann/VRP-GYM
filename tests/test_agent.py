import pytest
import torch

from agents import (
    IRPAgent,
    TSPAgent,
    VRPAgent,
    RandomAgent,
    GraphEncoder,
    GraphDemandEncoder,
    GraphDecoder,
)
from gym_vrp.envs import IRPEnv, TSPEnv, VRPEnv
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def setup():
    seed = 69
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_encoder():
    env = VRPEnv(num_nodes=8, batch_size=2, num_draw=1)
    state = torch.from_numpy(env.reset()).float()

    encoder = GraphEncoder(node_input_dim=2)
    emb = encoder(state[:, :, :2])

    assert emb.shape == (2, 8, 128)


def test_decoder():
    num_graphs = 2
    num_nodes = 8

    env = VRPEnv(num_nodes=num_nodes, batch_size=num_graphs, num_draw=1)

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

    assert torch.equal(next_node, torch.tensor([[5], [7]], dtype=torch.long))


def test_random_agent():
    num_graphs = 2
    num_nodes = 8

    env = VRPEnv(
        num_nodes=num_nodes,
        batch_size=num_graphs,
        num_draw=1,
    )
    agent = RandomAgent()
    loss = agent(env)

    assert np.isclose([loss.mean().item()], [-5.585874557495117])


def test_tsp_agent():
    num_graphs = 2
    num_nodes = 4

    env = TSPEnv(
        num_nodes=num_nodes,
        batch_size=num_graphs,
        num_draw=1,
    )
    agent = TSPAgent()
    loss, _, _ = agent.step(env, [True, True])

    assert np.isclose([loss.mean().item()], [-1.5130789279937744])


def test_vrp_agent():
    num_graphs = 2
    num_nodes = 4

    env = VRPEnv(
        num_nodes=num_nodes,
        batch_size=num_graphs,
        num_draw=1,
    )
    agent = VRPAgent()
    loss, _, _ = agent.step(env, [True, True])

    assert np.isclose([loss.mean().item()], [-1.952601671218872])


def test_irp_agent():
    num_graphs = 2
    num_nodes = 4

    env = IRPEnv(
        num_nodes=num_nodes,
        batch_size=num_graphs,
        num_draw=1,
    )
    agent = IRPAgent()
    loss, _, _ = agent.step(env, [True, True])

    assert np.isclose([loss.mean().item()], [-2.9770922660827637])
