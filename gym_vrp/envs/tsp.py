from typing import Tuple, Union

import numpy as np
from gym import Env
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from ..graph.vrp_network import VRPNetwork
from .common import ObsType


class TSPEnv(Env):
    """
    TSPEnv implements the Traveling Salesmen Problem
    a special variant of the vehicle routing problem.

    State: Shape (batch_size, num_nodes, 4) The third
        dimension is structured as follows:
        [x_coord, y_coord, is_depot, visitable]

    Actions: Depends on the number of nodes in every graph.
        Should contain the node numbers to visit next for
        each graph. Shape (batch_size, 1)
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_nodes: int = 20,
        batch_size: int = 128,
        num_draw: int = 6,
        seed: int = 69,
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
        assert (
            num_draw <= batch_size
        ), "Num_draw needs to be equal or lower than the number of generated graphs."

        np.random.seed(seed)

        self.step_count = 0
        self.num_nodes = num_nodes
        self.batch_size = batch_size

        # init video recorder
        self.draw_idxs = np.random.choice(batch_size, num_draw, replace=False)
        self.video_save_path = None

        self.generate_graphs()

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

        self.current_location = np.array(actions)

        if self.video_save_path is not None:
            self.vid.capture_frame()

        done = self.is_done()
        return (
            self.get_state(),
            -self.sampler.get_distances(traversed_edges),
            done,
            None,
        )

    def is_done(self):
        return np.all(self.visited == 1)

    def get_state(self) -> np.ndarray:
        """
        Getter for the current environment state

        Returns:
            np.ndarray: Shape (num_graph, num_nodes, 4)
            where the third dimension consists of the
            x, y coordinates, if the node is a depot,
            and if it has been visted yet.
        """

        # generate state (depots not yet set)
        state = np.dstack(
            [
                self.sampler.get_graph_positions(),
                np.zeros((self.batch_size, self.num_nodes)),
                self.generate_mask(),
            ]
        )

        # set depots in state to 1
        state[np.arange(len(state)), self.depots.T, 2] = 1

        return state

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

        # allow staying on a depot if the graph is solved.
        done_graphs = np.where(np.all(self.visited, axis=1) == True)[0]
        self.visited[done_graphs, self.depots[done_graphs].squeeze()] = 0

        return self.visited

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        """
        Resets the environment. 

        Returns:
            Union[ObsType, Tuple[ObsType, dict]]: State of the environment.
        """

        self.step_count = 0
        self.generate_graphs()
        return self.get_state()

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

    def render(self, mode: str = "human"):
        """
        Visualize one step in the env. Since its batched 
        this methods renders n random graphs from the batch.
        """
        return self.sampler.draw(self.draw_idxs)

    def enable_video_capturing(self, video_save_path: str):
        self.video_save_path = video_save_path
        if self.video_save_path is not None:
            self.vid = VideoRecorder(self, self.video_save_path)
            self.vid.frames_per_sec = 1
