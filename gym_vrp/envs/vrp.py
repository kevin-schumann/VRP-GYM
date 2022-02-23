from typing import Optional, Tuple, TypeVar, Union
from gym import Env
from enum import Enum, auto
import numpy as np

ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class VRPVariant(Enum):
    """ Type of VRP variant to initialise. """

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

    def __init__(self):
        ...

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """
        Run the environment one timestep. It's the users responsiblity to
        call reset() when the end of the episode has been reached. Accepts
        an actions and return a tuple of (observation, reward, done, info)

        Args:
            action (ActType): Action by the agent. Should be within the
                environments action space (self.action_space).

        Returns:
            Tuple[ObsType, float, bool, dict]: Tuple of the observation, 
                reward, done and info. The observation is within
                self.observation_space. The reward is for the previous action.
                If done equals True then the episode is over. Stepping through 
                environment while done returns undefined results. Info contains
                may contain additions info in terms of metrics, state variables
                and such.
        """
        ...

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

    def render(self, mode: str = "human") -> Optional[Union[np.ndarray, str]]:
        ...

    def close(self):
        ...
