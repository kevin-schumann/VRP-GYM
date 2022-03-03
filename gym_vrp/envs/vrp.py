from typing import Optional, Tuple, TypeVar, Union
from gym import Env
from enum import Enum, auto
import numpy as np
from ..graph.vrp_network import VRPNetwork
from gym.wrappers.monitoring.video_recorder import VideoRecorder

ObsType = TypeVar("ObsType")


class DefaultTSPEnv(Env):
    """
    This class implements the the default vehicle
    routing problem (vrp) in accordance to the the
    open-ai gym api as an environment.
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
        assert num_draw <= batch_size

        np.random.seed(seed)
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.step_count = 0
        self.draw_idxs = np.random.choice(batch_size, num_draw, replace=False)
        if video_path is not None:
            self.vid = VideoRecorder(self, video_path)
            self.vid.frames_per_sec = 1
        self.video_path = video_path

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
        assert actions.shape[0] == self.batch_size

        self.visited[np.arange(len(actions)), actions.T] = 1
        self.step_count += 1

        # walking steps in current state
        paths = np.hstack([self.prev_actions, actions]).astype(
            int
        )  # shape: batch_size x 2

        self.sampler.visit_edges(paths)
        self.prev_actions = np.array(actions)
        done = self.is_done()

        if self.video_path is not None:
            self.vid.capture_frame()

        return (
            self.get_state(),
            self.sampler.get_distances(paths),
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
        # set if each node is depot or not (1 means node is depot)
        state[np.arange(len(state)), self.depots.T, 2] = 1

        return state

    def generate_mask(self):
        depot_graphs_idxs = np.where(self.prev_actions == self.depots)[
            0
        ]  # doesnt work for multiple depots
        depot_graphs_idxs_not = np.where(self.prev_actions != self.depots)[0]
        self.visited[depot_graphs_idxs, self.depots[depot_graphs_idxs].squeeze()] = 1
        # self.visited[
        #     depot_graphs_idxs_not, self.depots[depot_graphs_idxs_not].squeeze()
        # ] = 0
        # make depot visitiable if all nodes are visited
        done_graphs = np.where(np.all(self.visited, axis=1) == True)[0]
        self.visited[done_graphs, self.depots[done_graphs].squeeze()] = 0

        return self.visited

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        """
        Reset the environment

        Args:
            seed (Optional[int], optional): _description_. Defaults to None.
            return_info (bool, optional): _description_. Defaults to False.
            options (Optional[dict], optional): _description_. Defaults to None.

        Returns:
            Union[ObsType, Tuple[ObsType, dict]]: _description_
        """

        self.step_count = 0
        self.generate_graphs()
        return self.get_state()

    def render(self, mode: str = "human"):
        """
        Visualize one step in the env. Since its batched this methods renders n random graphs
        from the batch.

        Args:
            mode (str): ...
        """
        return self.sampler.draw(self.draw_idxs)

    def close(self):
        ...

    def generate_graphs(self):
        self.visited = np.zeros(shape=(self.batch_size, self.num_nodes))
        self.sampler = VRPNetwork(
            num_graphs=self.batch_size, num_nodes=self.num_nodes, num_depots=1,
        )

        # Generate start points for each graph in batch
        self.depots = self.sampler.get_depots()
        self.prev_actions = (
            self.depots
        )  # self.depots[:, np.random.choice(self.depots.shape[1], 1)]


class DefaultVRPEnv(DefaultTSPEnv):
    def generate_mask(self):
        depot_graphs_idxs = np.where(self.prev_actions == self.depots)[
            0
        ]  # doesnt work for multiple depots
        depot_graphs_idxs_not = np.where(self.prev_actions != self.depots)[0]
        self.visited[depot_graphs_idxs, self.depots[depot_graphs_idxs].squeeze()] = 1
        self.visited[
            depot_graphs_idxs_not, self.depots[depot_graphs_idxs_not].squeeze()
        ] = 0
        # make depot visitiable if all nodes are visited
        done_graphs = np.where(np.all(self.visited, axis=1) == True)[0]
        self.visited[done_graphs, self.depots[done_graphs].squeeze()] = 0

        return self.visited


class DemandVRPEnv(DefaultTSPEnv):
    """
    This class implements the the vehicle
    routing problem (vrp) with demand on nodes and vehicles in accordance to the the
    open-ai gym api as an environment.
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
        assert num_draw <= batch_size

        np.random.seed(seed)
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.step_count = 0
        self.draw_idxs = np.random.choice(batch_size, num_draw, replace=False)
        self.load = np.ones(shape=(batch_size,))
        if video_path is not None:
            self.vid = VideoRecorder(self, video_path)
            self.vid.frames_per_sec = 1
        self.video_path = video_path

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
        assert actions.shape[0] == self.batch_size

        self.visited[np.arange(len(actions)), actions.T] = 1
        self.step_count += 1

        # walking steps in current state
        paths = np.hstack([self.prev_actions, actions]).astype(
            int
        )  # shape: batch_size x 2

        self.sampler.visit_edges(paths)

        selected_demands = self.demands[
            np.arange(len(self.demands)), actions.T
        ].squeeze()

        self.load -= selected_demands
        # self.load = np.where(self.load < 0, 0, self.load)
        # set load to 1 if on depot
        self.load[np.where(actions == self.depots)[0]] = 1

        self.prev_actions = np.array(actions)
        done = self.is_done()

        if self.video_path is not None:
            self.vid.capture_frame()

        return (
            self.get_state(),
            self.sampler.get_distances(paths),
            done,
            None,
        )

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
                self.demands,
                np.zeros((self.batch_size, self.num_nodes)),
                self.generate_mask(),
            ]
        )
        # set if each node is depot or not (1 means node is depot)
        state[np.arange(len(state)), self.depots.T, 3] = 1

        return state

    def generate_mask(self):
        # mask out the nodes that need to much demand
        depot_graphs_idxs = np.where(self.prev_actions == self.depots)[
            0
        ]  # doesnt work for multiple depots
        depot_graphs_idxs_not = np.where(self.prev_actions != self.depots)[0]
        self.visited[depot_graphs_idxs, self.depots[depot_graphs_idxs].squeeze()] = 1
        self.visited[
            depot_graphs_idxs_not, self.depots[depot_graphs_idxs_not].squeeze()
        ] = 0
        # make depot visitiable if all nodes are visited
        done_graphs = np.where(np.all(self.visited, axis=1) == True)[0]
        self.visited[done_graphs, self.depots[done_graphs].squeeze()] = 0

        mask = np.copy(self.visited)
        exceed_demand_idxs = ((self.demands - self.load[:, None, None]) > 0).squeeze()
        mask[exceed_demand_idxs] = 1

        return mask

    def generate_graphs(self):
        self.visited = np.zeros(shape=(self.batch_size, self.num_nodes))
        self.sampler = VRPNetwork(
            num_graphs=self.batch_size, num_nodes=self.num_nodes, num_depots=1,
        )

        # Generate start points for each graph in batch
        self.depots = self.sampler.get_depots()
        self.demands = self.sampler.get_demands()  # shape: (batch, node, 1)
        self.prev_actions = (
            self.depots
        )  # self.depots[:, np.random.choice(self.depots.shape[1], 1)]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        """
        Reset the environment

        Args:
            seed (Optional[int], optional): _description_. Defaults to None.
            return_info (bool, optional): _description_. Defaults to False.
            options (Optional[dict], optional): _description_. Defaults to None.

        Returns:
            Union[ObsType, Tuple[ObsType, dict]]: _description_
        """
        super().reset()
        self.load = np.ones(shape=(self.batch_size,))
        return self.get_state()
