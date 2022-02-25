import pytest
from gym_vrp.envs.vrp import DefaultVRPEnv
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def setup():
    pytest.node_num = 32
    pytest.graph_num = 4
    pytest.env = DefaultVRPEnv(pytest.node_num, pytest.graph_num, seed=69)


def test_init():
    assert len(pytest.env.sampler.graphs) == 4
    assert len(pytest.env.sampler.graphs[0].nodes) == 32


def test_step():
    actions = np.asarray([2, 2, 2, 2])[:, None]
    batched_reward = pytest.env.step(actions)

    assert batched_reward == -0.4289347094969973
