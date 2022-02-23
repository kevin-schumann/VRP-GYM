from dataclasses import dataclass
from typing import List
import networkx as nx


@dataclass
class VRPGraph:

    graph: nx.Graph = nx.Graph()

    @property
    def edges(self):
        return self.graph.edges.data()

    @property
    def nodes(self):
        return self.graph.nodes.data()


class VRPNetwork:
    def generate(num_graphs: int, min_nodes: int, max_nodes: int) -> List[VRPGraph]:
        """
        Generate graphs which are fully connected. This can be done by placing
        nodes randomly in an euclidean space and connecting all nodes with each other.
        """
        ...


class BaseNode:
    ...


class DepotNode(BaseNode):
    ...


class CustomerNode(BaseNode):
    ...

