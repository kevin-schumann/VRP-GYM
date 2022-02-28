import pytest
import torch
from gym_vrp.envs.vrp import DefaultVRPEnv
from agents.graph_encoder import GraphEncoder
from agents.graph_decoder import GraphDecoder


def test_encoder():
    env = DefaultVRPEnv(num_nodes=8, batch_size=2, num_draw=1)
    state = torch.from_numpy(env.reset()).float()
    depot_idx = torch.argmax(state[:, :, 2], axis=1)

    encoder = GraphEncoder(depot_input_dim=3, node_input_dim=2)
    emb = encoder(state[:, :, :2], depot_idx)

    assert emb.shape == (2, 8, 128)


def test_decoder():

    num_graphs = 2
    num_nodes = 8

    env = DefaultVRPEnv(num_nodes=num_nodes, batch_size=num_graphs, num_draw=1)

    # extract important state info
    state = torch.from_numpy(env.reset()).float()
    depot_idx = torch.argmax(state[:, :, 2], axis=1)

    encoder = GraphEncoder(depot_input_dim=3, node_input_dim=2)
    decoder = GraphDecoder(v_dim=128, k_dim=128)

    # encode
    embs = encoder(
        state[:, :, :2], depot_idx
    )  # shape -> (num_graphs, num_nodes, emb_size(128))

    # decode with initial mask
    mask = torch.zeros(size=(num_graphs, num_nodes))
    next_node = decoder(embs, mask)

    assert torch.equal(next_node, torch.tensor([[2], [6]], dtype=torch.long))
