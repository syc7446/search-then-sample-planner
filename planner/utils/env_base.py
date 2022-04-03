"""Base class for an environment.
"""

import abc
import numpy as np


class Environment:
    """Base class for an environment.
    """
    def __init__(self, num_objs, seed):
        self._num_objs = num_objs
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    @abc.abstractmethod
    def parse_state(self, state):
        """Parse the given state into literals.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_train_initial_states(self):
        """Returns a list of initial states for the training set.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def simulate(self, state, action):
        """Returns a next_state drawn from the transition model, along
        with a reward and done bit.
        """
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def literal_goal(self):
        """Get the goal expressed as a set of literals.
        """
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def discrete_predicates(self):
        """Get a set of all discrete (not continuous) predicates in this env.
        """
        raise NotImplementedError("Override me!")


class EnvironmentFailure(Exception):
    """Exception raised when something goes wrong in an environment.
    """
    pass
