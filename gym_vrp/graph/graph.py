from dataclasses import dataclass, field
from typing import List
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# TODO Move to util
@dataclass(frozen=True)
class NodeRange:
    min: int = 0
    max: int = 0


class VRPGraph:

    graph: nx.Graph = nx.Graph()

    def __init__(self, node_num: int, depot_num: int):
        """
        Creates a fully connected graph with node_num nodes
        and depot num depots. Coordinates of each node
        and the depot nodes will be samples randomly.

        Args:
            node_num (int): Number of nodes in the graph.
            depot_num (int): Number of depots in the graph.
        """

        self.graph = nx.complete_graph(node_num)

        # set coordinates for each node
        node_position = {
            i: coordinates for i, coordinates in enumerate(np.random.rand(node_num, 2))
        }
        nx.set_node_attributes(self.graph, node_position, "coordinates")

        # sample depots within the graph
        depots = np.random.choice(node_num, size=depot_num, replace=False)
        one_hot = np.zeros(node_num)
        one_hot[depots] = 1
        one_hot_dict = {i: depot for i, depot in enumerate(one_hot)}
        nx.set_node_attributes(self.graph, one_hot_dict, "depot")

    def draw(self):
        """
        Draws the graph as a matplotlib plot.
        Depots are drawn in red.
        """

        # Generate color for each node
        color_map = {0: "black", 1: "red"}
        node_colors = [
            color_map[node[1]["depot"]] for node in self.graph.nodes(data=True)
        ]
        pos = nx.get_node_attributes(self.graph, "coordinates")

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, edge_color="grey")

        plt.show()

    @property
    def edges(self):
        return self.graph.edges.data()

    @property
    def nodes(self):
        return self.graph.nodes.data()


class VRPNetwork:
    def __init__(
        self,
        num_graphs: int,
        num_nodes: NodeRange,
        num_depots: NodeRange,
    ) -> List[VRPGraph]:
        """
        Generate graphs which are fully connected. This can be done by placing
        nodes randomly in an euclidean space and connecting all nodes with each other.
        """
        assert (
            num_nodes.min >= num_depots.max
        ), "Number of depots should be lower than number of depots"

        self.graphs: List[VRPGraph] = []

        # sample number of nodes / depots
        node_nums = np.random.randint(
            low=num_nodes.min, high=num_nodes.max + 1, size=(num_graphs,)
        )
        depot_nums = np.random.randint(
            low=num_depots.min, high=num_depots.max + 1, size=(num_graphs,)
        )

        # generate a graph with nn nodes and nd depots
        for nn, nd in zip(node_nums, depot_nums):
            self.graphs.append(
                VRPGraph(
                    nn,
                    nd,
                )
            )
