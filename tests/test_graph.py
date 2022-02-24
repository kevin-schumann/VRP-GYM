from gym_vrp.graph.graph import VRPGraph, VRPNetwork, NodeRange
import networkx as nx
import numpy as np
import pytest


@pytest.fixture(scope="function", autouse=True)
def setup():
    np.random.seed(0)


def test_vrp_graph_init():
    """
    Inits 1 graph with 10 nodes and 5 depots.
    Checks if the graphs has 10 nodes and 5 depots.
    """
    graph = VRPGraph(10, 5)
    depots = list(nx.get_node_attributes(graph.graph, "depot").values())

    assert len(graph.nodes) == 10
    assert depots.count(1) == 5


def test_vrp_graph_network_init():
    """
    Initialises VRPNetwork with 10 graphs
    with 5-10 nodes and 1-2 depots each.

    Tests if we actually create 10 different
    graphs.
    """

    num_graphs = 10
    num_nodes: NodeRange = NodeRange(5, 10)
    num_depots: NodeRange = NodeRange(1, 2)
    network: VRPNetwork = VRPNetwork(
        num_graphs=num_graphs, num_nodes=num_nodes, num_depots=num_depots
    )
    assert len(set(num_nodes_each_graph)) != 1

    # get number of depots for each graph
    num_depots_each_graph = []
    for graph in network.graphs:
        depots = list(nx.get_node_attributes(graph.graph, "depot").values())
        num_depots_each_graph.append(depots.count(1))
    assert len(set(num_depots_each_graph)) != 1

    # get number of nodes for each graph
    num_nodes_each_graph = [len(graph.nodes) for graph in network.graphs]
    assert len(network.graphs) == num_graphs
