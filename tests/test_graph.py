from gym_vrp.graph.graph import VRPGraph, VRPNetwork, NodeRange
import networkx as nx
import numpy as np
import pytest


@pytest.fixture(scope="function", autouse=True)
def setup():
    seed = 69
    np.random.seed(seed)


def test_vrp_graph_init():
    """
    Inits 1 graph with 10 nodes and 5 depots.
    Checks if the graphs has 10 nodes and 5 depots.
    """
    graph = VRPGraph(10, 5)
    depots = list(nx.get_node_attributes(graph.graph, "depot").values())

    assert len(graph.nodes) == 10
    assert depots.count(1) == 5


def test_vrp_euclid_dist():
    """
    Test calc of euclid distance between
    two points in vrp graph.
    """

    graph = VRPGraph(2, 1)

    # change coordiantes s.t.
    # euclid distance will be 5.
    custom_coordinates = {
        0: np.array([2, -1]),
        1: np.array([-2, 2]),
    }
    nx.set_node_attributes(graph, custom_coordinates, "coordinates")

    assert graph.euclid_distance(0, 1) == 5


def test_vrp_graph_network_init():
    """
    Initialises VRPNetwork with 10 graphs
    with 10 nodes and 2 depots each.

    Tests if we actually create different
    graphs. Meaning we have different depots
    and coordinates etc.
    """

    num_graphs = 10
    num_nodes: int = 10
    num_depots: int = 2
    network: VRPNetwork = VRPNetwork(
        num_graphs=num_graphs, num_nodes=num_nodes, num_depots=num_depots
    )
    assert len(network.graphs) == num_graphs

    depots = network.get_depots()
    assert depots.shape == (num_graphs, num_depots)
    assert np.unique(depots, axis=0).shape[0] > 1
