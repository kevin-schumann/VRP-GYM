import numpy as np

from .tsp import TSPEnv


class VRPEnv(TSPEnv):
    """
    Implements the Vehicle Routing Problem as an gym 
    environment. Is similar to the Traveling Salesman
    Problem, just the depots are repeatable visitable.
    """

    def generate_mask(self):
        """
        Generates a mask of where the nodes marked as 1 cannot 
        be visited in the next step according to the env dynamic.

        Returns:
            np.ndarray: Returns mask for each (un)visitable node 
                in each graph. Shape (batch_size, num_nodes)
        """

        # disallow staying on a depot
        depot_graphs_idxs = np.where(self.current_location == self.depots)[0]
        self.visited[depot_graphs_idxs, self.depots[depot_graphs_idxs].squeeze()] = 1

        # allow visiting the depot when not currently at the depot
        depot_graphs_idxs_not = np.where(self.current_location != self.depots)[0]
        self.visited[
            depot_graphs_idxs_not, self.depots[depot_graphs_idxs_not].squeeze()
        ] = 0

        # allow staying on a depot if the graph is solved.
        done_graphs = np.where(np.all(self.visited, axis=1) == True)[0]
        self.visited[done_graphs, self.depots[done_graphs].squeeze()] = 0

        return self.visited
