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
        self.depots = np.random.choice(node_num, size=depot_num, replace=False)

        # set depots as attribute in nodes
        one_hot = np.zeros(node_num)
        one_hot[self.depots] = 1
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

    def euclid_distance(self, node1_idx: int, node2_idx: int) -> float:
        """
        Calculates the euclid distance between two nodes
        with their idx's respectively.
        """

        node_one_pos = self.graph.nodes[node1_idx]["coordinates"]
        node_two_pos = self.graph.nodes[node2_idx]["coordinates"]

        return np.linalg.norm(node_one_pos - node_two_pos)


class VRPNetwork:
    def __init__(
        self,
        num_graphs: int,
        num_nodes: int,
        num_depots: int,
    ) -> List[VRPGraph]:
        """
        Generate graphs which are fully connected. This can be done by placing
        nodes randomly in an euclidean space and connecting all nodes with each other.
        """

        assert (
            num_nodes >= num_depots
        ), "Number of depots should be lower than number of depots"

        self.num_nodes = num_nodes
        self.num_depots = num_depots
        self.num_graphs = num_graphs
        self.graphs: List[VRPGraph] = []

        # generate a graph with nn nodes and nd depots

        for _ in range(num_graphs):
            self.graphs.append(
                VRPGraph(
                    num_nodes,
                    num_depots,
                )
            )

    def get_distance(self, graph_idx: int, node_idx_1: int, node_idx_2: int) -> float:
        return self.graphs[graph_idx].euclid_distance(node_idx_1, node_idx_2)

    def get_distances(self, paths) -> List[float]:
        """
        Calculatest the euclid distance between
        each node pair in paths.

        Args:
            paths (nd.array): Shape num_graphs x 2
                where the second dimension denotes
                [source_node, target_node].

        Returns:
            List[float]: Euclid distance between each
                node pair.
        """

        return [self.get_distance(index, item[0], item[1]) for index, item in paths]

    def get_depots(self) -> List[List[int]]:
        """
        Get the depots of every graph within the network.

        Returns:
            List[List[int]]: Returns nd.array of shape
                num_graphs x num_depots.
        """

        depos_idx = np.zeros((self.num_graphs, self.num_depots))

        for i in range(self.num_graphs):
            depos_idx[i] = self.graphs[i].depots

        return depos_idx
