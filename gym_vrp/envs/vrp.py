from typing import Optional, Tuple, TypeVar, Union
from gym import Env
from enum import Enum, auto
import numpy as np
from ..graph.graph import VRPNetwork, NodeRange

ObsType = TypeVar("ObsType")


class VRPVariant(Enum):
    """Type of VRP variant to initialise."""

    DEFAULT_VRP = auto()


class VRPEnv:
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """
        Registers every subclass in the subclass-dict.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.variant] = cls

    @classmethod
    def create(cls, variant: VRPVariant, **kwargs):
        """
        Creates subclass depending on typ.

        Args:
            typ: VRPVariant
        Returns:
            Subclass that is used for the given VRPVariant.
        """
        if variant not in cls.subclasses:
            raise ValueError("Bad message type {}".format(variant))

        return cls.subclasses[variant]()


class DefaultVRPEnv(VRPEnv, Env):
    """
    This class implements the the default vehicle
    routing problem (vrp) in accordance to the the
    open-ai gym api as an environment.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}
    variant: VRPVariant = VRPVariant.DEFAULT_VRP

    def __init__(
        self,
        num_nodes: int = 32,
        batch_size: int = 128,
        num_draw: int = 6,
        seed: int = 69,
    ):
        assert num_draw <= batch_size

        np.random.seed(seed)
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.step_count = 0
        self.draw_idxs = np.random.choice(batch_size, num_draw, replace=False)

        self.__generate_graphs()

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

        self.sampler.color_edges(paths)
        self.prev_actions = np.array(actions)
        done = self.__is_done()

        return (
            self.get_state(),
            self.sampler.get_distances(paths),
            done,
            None,
        )

    def __is_done(self):
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
                self.__generate_mask(),
            ]
        )
        # set if each node is depot or not (1 means node is depot)
        state[np.arange(len(state)), self.depots.T, 2] = 1

        return state

    def __generate_mask(self):
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
        self.__generate_graphs()
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

    def __generate_graphs(self):
        self.visited = np.zeros(shape=(self.batch_size, self.num_nodes))
        self.sampler = VRPNetwork(
            num_graphs=self.batch_size, num_nodes=self.num_nodes, num_depots=1,
        )

        # Generate start points for each graph in batch
        self.depots = self.sampler.get_depots()
        self.prev_actions = (
            self.depots
        )  # self.depots[:, np.random.choice(self.depots.shape[1], 1)]
