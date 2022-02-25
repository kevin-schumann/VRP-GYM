from dataclasses import dataclass, field
from typing import List
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# TODO Move to util
@dataclass(frozen=True)
class NodeRange:
    min: int = 0
    max: int = 0


class VRPGraph:

    graph: nx.Graph = nx.Graph()

    def __init__(self, num_nodes: int, num_depots: int):
        """
        Creates a fully connected graph with node_num nodes
        and depot num depots. Coordinates of each node
        and the depot nodes will be samples randomly.

        Args:
            node_num (int): Number of nodes in the graph.
            depot_num (int): Number of depots in the graph.
        """
        self.num_nodes = num_nodes
        self.num_depots = num_depots
        self.graph = nx.complete_graph(num_nodes)

        # set coordinates for each node
        node_position = {
            i: coordinates for i, coordinates in enumerate(np.random.rand(num_nodes, 2))
        }
        nx.set_node_attributes(self.graph, node_position, "coordinates")

        # sample depots within the graph
        self.depots = np.random.choice(num_nodes, size=num_depots, replace=False)

        # set depots as attribute in nodes
        one_hot = np.zeros(num_nodes)
        one_hot[self.depots] = 1
        one_hot_dict = {i: depot for i, depot in enumerate(one_hot)}
        nx.set_node_attributes(self.graph, one_hot_dict, "depot")

        self._set_default_colors()

    def _set_default_colors(self):
        """
        Sets the default colors of the edges / nodes
        as attributes. Color schema is:

        Node is black except when it is a depot which
        shall be red. Unused edges are grey els red.
        """
        nx.set_edge_attributes(self.graph, "grey", "edge_color")
        nx.set_node_attributes(self.graph, "black", "node_color")

        for node in self.depots:
            self.graph.nodes[node]["node_color"] = "red"

    def draw(self, ax):
        """
        Draws the graph as a matplotlib plot.
        Depots are drawn in red.
        """

        # draw nodes according to color and position attribute
        pos = nx.get_node_attributes(self.graph, "coordinates")
        node_colors = nx.get_node_attributes(self.graph, "node_color").values()
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, ax=ax)

        # draw edges
        edge_colors = nx.get_edge_attributes(self.graph, "edge_color").values()
        nx.draw_networkx_edges(
            self.graph, pos, alpha=0.5, edge_color=edge_colors, ax=ax
        )

    def color_edge(self, source_node: int, target_node: int) -> None:
        """
        Sets the edge color to red.

        Args:
            source_node (int): Source node of the edge
            target_node (int): Target node of the edge
        """
        self.graph.edges[source_node, target_node]["edge_color"] = "red"

    @property
    def edges(self):
        return self.graph.edges.data()

    @property
    def nodes(self):
        return self.graph.nodes.data()

    @property
    def get_node_positions(self) -> np.ndarray:
        """
        Returns the coordinates of each node as
        an ndarray of shape (num_nodes, 2) sorted
        by the node index.
        """

        positions = nx.get_node_attributes(self.graph, "coordinates").values()
        return np.asarray(positions).reshape(self.num_nodes, 2)

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
        return [
            self.get_distance(index, source, dest)
            for index, (source, dest) in enumerate(paths)
        ]

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

    def draw(self, graph_idxs: List[int]) -> None:
        """
        Draw multiple graphs in a matplotlib grid.

        Args:
            graph_idxs (List[int]): Idxs of graphs which get drawn.
        """

        num_columns = min(len(graph_idxs), 3)
        num_rows = np.ceil(len(graph_idxs) / num_columns).astype(int)

        plt.clf()
        fig = plt.figure(figsize=(5 * num_columns, 5 * num_rows))

        for n, graph_idx in enumerate(graph_idxs):
            ax = plt.subplot(num_rows, num_columns, n + 1)

            self.graphs[graph_idx].draw(ax=ax)
        plt.show()

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image

    def color_edges(self, transition_matrix: List[List[int]]) -> None:
        """
        Colors the mentioned edge for each graph.

        Args:
            transition_matrix (List[List[int]]): Shape num_graphs x 2
                where each row is [source_node_idx, target_node_idx].
        """
        for i, row in enumerate(transition_matrix):
            self.graphs[i].color_edge(row[0], row[1])

    def get_graph_positions(self) -> np.ndarray:
        """
        Returns the coordinates of each node in every graph as
        an ndarray of shape (num_graphs, num_nodes, 2) sorted
        by the graph and node index.
        """

        node_positions = np.zeros(shape=(len(self.graphs), self.num_nodes, 2))
        for i, graph in enumerate(self.graphs):
            node_positions[i] = graph.get_node_positions()

        return node_positions
