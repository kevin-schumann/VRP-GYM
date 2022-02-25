import pytest
from gym_vrp.envs.vrp import DefaultVRPEnv
import numpy as np
import networkx as nx
import math


@pytest.fixture(scope="session", autouse=True)
def setup():
    pytest.node_num = 3
    pytest.graph_num = 2
    pytest.env = DefaultVRPEnv(pytest.node_num, pytest.graph_num, 2, seed=69)

    y_coord = math.sqrt(3) / 2
    graph_one_coords = {
        0: np.array([0, 0]),
        1: np.array([1, 0]),
        2: np.array([0.5, y_coord]),
    }

    graph_two_coords = {
        0: np.array([0, 0]),
        1: np.array([4, 0]),
        2: np.array([2, 4 * y_coord]),
    }

    nx.set_node_attributes(
        pytest.env.sampler.graphs[0], graph_one_coords, "coordinates"
    )
    nx.set_node_attributes(
        pytest.env.sampler.graphs[1], graph_two_coords, "coordinates"
    )


def test_init():
    assert len(pytest.env.sampler.graphs) == 2
    assert len(pytest.env.sampler.graphs[0].nodes) == 3


def test_step():
    actions = np.asarray([2, 2])[:, None]
    batched_reward = pytest.env.step(actions)

    assert np.isclose(batched_reward, -(1 + 4) / 2)
