# from typing import List

# import matplotlib.pyplot as plt
# import numpy as np


# class VRPNetwork:
#     def __init__(
#         self, num_graphs: int, num_nodes: int, num_depots: int,
#     ) -> List[VRPGraph]:
#         """
#         Creates num_graphs random generated fully connected
#         graphs with num_nodes nodes. Node positions are
#         sampled uniformly in [0, 1]. In each graph
#         num_debots nodes are marked as depots.

#         Args:
#             num_graphs (int): Number of graphs to generate.
#             num_nodes (int): Number of nodes in each graph.
#             num_depots (int): Number of depots in each graph.

#         Returns:
#             List[VRPGraph]: List of num_graphs networkx graphs
#         """

#         assert (
#             num_nodes >= num_depots
#         ), "Number of depots should be lower than number of depots"

#         self.num_nodes = num_nodes
#         self.num_depots = num_depots
#         self.num_graphs = num_graphs
#         self.graphs: List[VRPGraph] = []

#         # generate a graph with nn nodes and nd depots
#         for _ in range(num_graphs):
#             self.graphs.append(VRPGraph(num_nodes, num_depots,))

#     def get_distance(self, graph_idx: int, node_idx_1: int, node_idx_2: int) -> float:
#         """
#         Calculates the euclid distance between the two nodes
#         within a single graph in the VRPNetwork.

#         Args:
#             graph_idx (int): Index of the graph
#             node_idx_1 (int): Source node
#             node_idx_2 (int): Target node

#         Returns:
#             float: Euclid distance between the two nodes
#         """
#         return self.graphs[graph_idx].euclid_distance(node_idx_1, node_idx_2)

#     def get_distances(self, paths) -> np.ndarray:
#         """
#         Calculatest the euclid distance between
#         each node pair in paths.

#         Args:
#             paths (nd.array): Shape num_graphs x 2
#                 where the second dimension denotes
#                 [source_node, target_node].

#         Returns:
#             np.ndarray: Euclid distance between each
#                 node pair. Shape (num_graphs,)
#         """
#         return np.array(
#             [
#                 self.get_distance(index, source, dest)
#                 for index, (source, dest) in enumerate(paths)
#             ]
#         )

#     def get_depots(self) -> np.ndarray:
#         """
#         Get the depots of every graph within the network.

#         Returns:
#             np.ndarray: Returns nd.array of shape
#                 (num_graphs, num_depots).
#         """

#         depos_idx = np.zeros((self.num_graphs, self.num_depots), dtype=int)

#         for i in range(self.num_graphs):
#             depos_idx[i] = self.graphs[i].depots

#         return depos_idx

#     def get_demands(self) -> np.ndarray:
#         """
#         Returns the demands for each node in each graph.

#         Returns:
#             np.ndarray: Demands of each node in shape
#                 (num_graphs, num_nodes, 1)
#         """
#         demands = np.zeros(shape=(self.num_graphs, self.num_nodes, 1))
#         for i in range(self.num_graphs):
#             demands[i] = self.graphs[i].demand

#         return demands

#     def draw(self, graph_idxs: np.ndarray) -> None:
#         """
#         Draw multiple graphs in a matplotlib grid.

#         Args:
#             graph_idxs (np.ndarray): Idxs of graphs which get drawn.
#                 Expected to be of shape (x, ).

#         Returns:
#             np.ndarray: Plot as rgb-array of shape (width, height, 3).
#         """

#         num_columns = min(len(graph_idxs), 3)
#         num_rows = np.ceil(len(graph_idxs) / num_columns).astype(int)

#         # plot each graph in a 3 x num_rows grid
#         plt.clf()
#         fig = plt.figure(figsize=(5 * num_columns, 5 * num_rows))

#         for n, graph_idx in enumerate(graph_idxs):
#             ax = plt.subplot(num_rows, num_columns, n + 1)

#             self.graphs[graph_idx].draw(ax=ax)

#         plt.show()

#         # convert to plot to rgb-array
#         fig.canvas.draw()
#         data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

#         return image

#     def visit_edges(self, transition_matrix: np.ndarray) -> None:
#         """
#         Visits each edges specified in the transition matrix.

#         Args:
#             transition_matrix (np.ndarray): Shape num_graphs x 2
#                 where each row is [source_node_idx, target_node_idx].
#         """
#         for i, row in enumerate(transition_matrix):
#             self.graphs[i].visit_edge(row[0], row[1])

#     def get_graph_positions(self) -> np.ndarray:
#         """
#         Returns the coordinates of each node in every graph as
#         an ndarray of shape (num_graphs, num_nodes, 2) sorted
#         by the graph and node index.

#         Returns:
#             np.ndarray: Node coordinates of each graph. Shape
#                 (num_graphs, num_nodes, 2)
#         """

#         node_positions = np.zeros(shape=(len(self.graphs), self.num_nodes, 2))
#         for i, graph in enumerate(self.graphs):
#             node_positions[i] = graph.node_positions

#         return node_positions
