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

    metadata = {"render.modes": ["human"]}
    variant: VRPVariant = VRPVariant.DEFAULT_VRP

    def __init__(self, num_nodes: int = 32, batch_size: int = 128, seed: int = 69):
        np.random.seed(seed)
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.step_count = 0

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
        self.visited[:, actions] = 1
        self.step_count += 1

        # walking steps in current state
        paths = np.hstack([self.prev_action, actions])

        self.prev_action = actions
        return -np.mean(
            self.sampler.get_distances(paths), axis=0
        )  # return average negative walk length

    def is_done(self):
        return np.all(self.visited == 1)

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

        # TODO
        return None

    def render(self, mode: str = "human") -> Optional[Union[np.ndarray, str]]:
        ...

    def close(self):
        ...

    def __generate_graphs(self):
        self.visited = np.zeros(shape=(self.batch_size, self.num_nodes))
        self.sampler = VRPNetwork(
            num_graphs=self.batch_size,
            num_nodes=self.num_nodes,
            num_depots=1,
        )

        # Generate start points for each graph in batch
        depots = self.sampler.get_depots()
        self.prev_action = depots[:, np.random.choice(depots.shape[1], 1)]
