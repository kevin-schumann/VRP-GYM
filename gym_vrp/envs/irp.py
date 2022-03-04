from typing import Tuple, Union

import numpy as np

from ..graph.vrp_network import VRPNetwork
from .common import ObsType
from .tsp import TSPEnv


class IRPEnv(TSPEnv):
    """
    IRPEnv implements the Inventory Routing Problem a variant
    of the Vehicle Routing Problem. The vehicle has a
    capacity of 1. Visiting a node is only allowed if the
    cars capacity is greater or equal than the nodes demand.

    State: Shape (batch_size, num_nodes, 5) The third
        dimension is structured as follows:
        [x_coord, y_coord, demand, is_depot, visitable]

    Actions: Depends on the number of nodes in every graph.
        Should contain the node numbers to visit next for
        each graph. Shape (batch_size, 1)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_nodes: int = 32,
        batch_size: int = 128,
        num_draw: int = 6,
        seed: int = 69,
        video_path: str = None,
    ):
        """
        Args:
            num_nodes (int, optional): Number of nodes in each generated graph. Defaults to 32.
            batch_size (int, optional): Number of graphs to generate. Defaults to 128.
            num_draw (int, optional): When calling the render num_draw graphs will be rendered. 
                Defaults to 6.
            seed (int, optional): Seed of the environment. Defaults to 69.
            video_save_path (str, optional): When set a video of the interactions with the 
                environment is saved at the set location. Defaults to None.
        """
        super().__init__(
            num_nodes=num_nodes,
            batch_size=batch_size,
            num_draw=num_draw,
            seed=seed,
            video_save_path=video_path,
        )

        self.load = np.ones(shape=(batch_size,))

    def step(self, actions: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Run the environment one timestep. It's the users responsiblity to
        call reset() when the end of the episode has been reached. Accepts
        an actions and return a tuple of (observation, reward, done, info)

        Args:
            actions (nd.ndarray): Which node to visit for each graph.
                Shape of actions is (batch_size, 1).

        Returns:
            Tuple[ObsType, float, bool, dict]: Tuple of the observation,
                reward, done and info. The observation is within
                self.observation_space. The reward is for the previous action.
                If done equals True then the episode is over. Stepping through
                environment while done returns undefined results. Info contains
                may contain additions info in terms of metrics, state variables
                and such.
        """
        assert (
            actions.shape[0] == self.batch_size
        ), "Number of actions need to equal the number of generated graphs."

        self.step_count += 1

        # visit each next node
        self.visited[np.arange(len(actions)), actions.T] = 1
        traversed_edges = np.hstack([self.current_location, actions]).astype(int)
        self.sampler.visit_edges(traversed_edges)

        # get demand of the visited nodes
        selected_demands = self.demands[
            np.arange(len(self.demands)), actions.T
        ].squeeze()

        # update load of each vehicle
        self.load -= selected_demands
        self.load[np.where(actions == self.depots)[0]] = 1

        self.current_location = np.array(actions)

        if self.video_path is not None:
            self.vid.capture_frame()

        done = self.is_done()
        return (
            self.get_state(),
            self.sampler.get_distances(traversed_edges),
            done,
            None,
        )

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Getter for the current environment state.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Shape (num_graph, num_nodes, 5)
                The third dimension is structured as follows:
            [x_coord, y_coord, demand, is_depot, visitable]
        """

        # generate state (depots not yet set)
        state = np.dstack(
            [
                self.sampler.get_graph_positions(),
                self.demands,
                np.zeros((self.batch_size, self.num_nodes)),
                self.generate_mask(),
            ]
        )

        # set depots in state to 1
        state[np.arange(len(state)), self.depots.T, 3] = 1

        return (state, self.load)

    def generate_mask(self):
        """
        Generates a mask of where the nodes marked as 1 cannot
        be visited in the next step according to the env dynamic.

        Returns:
            np.ndarray: Returns mask for each (un)visitable node
                in each graph. Shape (batch_size, num_nodes)
        """

        # disallow staying at the depot
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

        # disallow visiting nodes that exceed the current load.
        mask = np.copy(self.visited)
        exceed_demand_idxs = ((self.demands - self.load[:, None, None]) > 0).squeeze()
        mask[exceed_demand_idxs] = 1

        return mask

    def generate_graphs(self):
        """
        Generates a VRPNetwork of batch_size graphs with num_nodes
        each. Resets the visited nodes to 0.
        """
        self.visited = np.zeros(shape=(self.batch_size, self.num_nodes))
        self.sampler = VRPNetwork(
            num_graphs=self.batch_size, num_nodes=self.num_nodes, num_depots=1,
        )

        # set current location to the depots
        self.depots = self.sampler.get_depots()
        self.current_location = self.depots

        self.demands = self.sampler.get_demands()

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        """
        Resets the environment.

        Returns:
            Union[ObsType, Tuple[ObsType, dict]]: State of the environment.
        """
        super().reset()
        self.load = np.ones(shape=(self.batch_size,))
        return self.get_state()
