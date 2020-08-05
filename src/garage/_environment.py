"""Base Garage Environment API."""

import abc
import collections
import math

import akro
import numpy as np

from garage import InOutSpec, StepType


class EnvSpec(InOutSpec):
    """Describes the action and observation spaces of an environment.

    Args:
        observation_space (akro.Space): The observation space of the env.
        action_space (akro.Space): The action space of the env.
        max_episode_length (int): The maximum number of steps allowed in an
            episode.

    """

    def __init__(self,
                 observation_space,
                 action_space,
                 max_episode_length=math.inf):
        self._max_episode_length = max_episode_length
        super().__init__(action_space, observation_space)

    @property
    def action_space(self):
        """Get action space.

        Returns:
            akro.Space: Action space of the env.

        """
        return self.input_space

    @property
    def observation_space(self):
        """Get observation space of the env.

        Returns:
            akro.Space: Observation space.

        """
        return self.output_space

    @action_space.setter
    def action_space(self, action_space):
        """Set action space of the env.

        Args:
            action_space (akro.Space): Action space.

        """
        self._input_space = action_space

    @observation_space.setter
    def observation_space(self, observation_space):
        """Set observation space of the env.

        Args:
            observation_space (akro.Space): Observation space.

        """
        self._output_space = observation_space

    @property
    def max_episode_length(self):
        """Get max episode steps.

        Returns:
            int: The maximum number of steps that an episode

        """
        return self._max_episode_length


class EnvStep(
        collections.namedtuple('EnvStep', [
            'env_spec', 'observation', 'action', 'reward', 'next_observation',
            'env_info', 'step_type'
        ])):
    # pylint: disable=missing-return-doc, missing-return-type-doc, missing-param-doc, missing-type-doc  # noqa: E501
    r"""A tuple representing a single step returned by the environment.

    Attributes:
        env_spec (garage.envs.EnvSpec): Specification for the environment from
            which this data was sampled.
        observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for the this time step in the
            environment. These must conform to
            :obj:`env_spec.observation_space`.
            The observation before applying the action.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        action (numpy.ndarray): A numpy array of shape :math:`(A^*)`
            containing the action for the this time step. These must conform
            to :obj:`env_spec.action_space`.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        reward (float): A float representing the reward for taking the action
            given the observation, at the this time step.
            `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
            sequence.
        next_observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
            containing the observation for the this time step in the
            environment. These must conform to
            :obj:`env_spec.observation_space`.
            The observation after applying the action.
        env_info (dict): A dict containing environment state information.
        step_type (StepType): a `StepType` enum value. Can either be
            StepType.FIRST, StepType.MID, StepType.TERMINAL, StepType.TIMEOUT.


    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """

    def __new__(cls, env_spec, observation, action, reward, next_observation,
                env_info, step_type):  # noqa: D102
        # pylint: disable=too-many-branches
        # observation
        if not env_spec.observation_space.contains(observation):
            if isinstance(env_spec.observation_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        observation.shape):
                    raise ValueError('observation should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         observation.shape))
            else:
                raise ValueError(
                    'observation must conform to observation_space {}, '
                    'but got data with shape {} instead.'.format(
                        env_spec.observation_space, observation))

        if not env_spec.observation_space.contains(next_observation):
            if isinstance(env_spec.observation_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        next_observation.shape):
                    raise ValueError('next_observation should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         next_observation.shape))
            else:
                raise ValueError(
                    'next_observation must conform to observation_space {}, '
                    'but got data with shape {} instead.'.format(
                        env_spec.observation_space, next_observation))

        # action
        if not env_spec.action_space.contains(action):
            if isinstance(env_spec.action_space,
                          (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.action_space.flat_dim != np.prod(action.shape):
                    raise ValueError('action should have the same '
                                     'dimensionality as the action_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.action_space.flat_dim,
                                         action.shape))
            else:
                raise ValueError('action must conform to action_space {}, '
                                 'but got data with shape {} instead.'.format(
                                     env_spec.action_space, action))

        if not isinstance(env_info, dict):
            raise ValueError('env_info must be type {}, but got type {} '
                             'instead.'.format(dict, type(env_info)))

        if not isinstance(reward, float):
            raise ValueError('reward must be type {}, but got type {} '
                             'instead.'.format(float, type(reward)))

        if not isinstance(step_type, StepType):
            raise ValueError(
                'step_type must be dtype garage.StepType, but got dtype {} '
                'instead.'.format(type(step_type)))

        return super().__new__(EnvStep, env_spec, observation, action, reward,
                               next_observation, env_info, step_type)

    @property
    def first(self):
        """bool: Whether this `TimeStep` is the first of a sequence."""
        return self.step_type is StepType.FIRST

    @property
    def mid(self):
        """bool: Whether this `TimeStep` is in the mid of a sequence."""
        return self.step_type is StepType.MID

    @property
    def terminal(self):
        """bool: Whether this `TimeStep` records a termination condition."""
        return self.step_type is StepType.TERMINAL

    @property
    def timeout(self):
        """bool: Whether this `TimeStep` records a time out condition."""
        return self.step_type is StepType.TIMEOUT

    @property
    def last(self):
        """bool: Whether this `TimeStep` is the last of a sequence."""
        return self.step_type is StepType.TERMINAL or self.step_type \
            is StepType.TIMEOUT


class Environment(abc.ABC):
    """The main API for garage environments.

    The public API methods are:

    +-----------------------+
    | Functions             |
    +=======================+
    | reset()               |
    +-----------------------+
    | step()                |
    +-----------------------+
    | render()              |
    +-----------------------+
    | visualize()           |
    +-----------------------+
    | close()               |
    +-----------------------+

    Set the following properties:

    +-----------------------+-------------------------------------------------+
    | Properties            | Description                                     |
    +=======================+=================================================+
    | action_space          | The action space specification                  |
    +-----------------------+-------------------------------------------------+
    | observation_space     | The observation space specification             |
    +-----------------------+-------------------------------------------------+
    | spec                  | The environment specifications                  |
    +-----------------------+-------------------------------------------------+
    | render_modes          | The list of supported render modes              |
    +-----------------------+-------------------------------------------------+

    Example of a simple rollout loop:

    .. code-block:: python

        env = MyEnv()
        policy = MyPolicy()
        first_observation, episode_info = env.reset()
        env.visualize()  # visualization window opened

        episode = []
        # Determine the first action
        first_action = policy.get_action(first_observation, episode_info)
        episode.append(env.step(first_action))

        while not episode[-1].last():
           action = policy.get_action(episode[-1].next_observation)
           episode.append(env.step(action))

        env.close()  # visualization window closed

    Make sure your environment is pickle-able:
        Garage pickles the environment via the `cloudpickle` module
        to save snapshots of the experiment. However, some environments may
        contain attributes that are not pickle-able (e.g. a client-server
        connection). In such cases, override `__setstate__()` and
        `__getstate__()` to add your custom pickle logic.

        You might want to refer to the EzPickle module:
        https://github.com/openai/gym/blob/master/gym/utils/ezpickle.py
        for a lightweight way of pickle and unpickle via constructor
        arguments.

    """

    @property
    @abc.abstractmethod
    def action_space(self):
        """akro.Space: The action space specification."""

    @property
    @abc.abstractmethod
    def observation_space(self):
        """akro.Space: The observation space specification."""

    @property
    @abc.abstractmethod
    def spec(self):
        """EnvSpec: The environment specification."""

    @property
    @abc.abstractmethod
    def render_modes(self):
        """list: A list of string representing the supported render modes.

        See render() for a list of modes.
        """

    @abc.abstractmethod
    def reset(self):
        """Resets the environment.

        Returns:
            numpy.ndarray: The first observation. It must conform to
                `observation_space`.
            dict: The episode-level information. Note that this is not part
                of `env_info` provided in `step()`. It contains information of
                the entire episode， which could be needed to determine the
                first action (e.g. in the case of goal-conditioned or MTRL.)

        """

    @abc.abstractmethod
    def step(self, action):
        """Steps the environment with the action and returns a `EnvStep`.

        If the environment returned the last `EnvStep` of a sequence (either
        of type TERMINAL or TIMEOUT) at the previous step, this call to
        `step()` will start a new sequence and `action` will be ignored.

        If `spec.max_episode_length` is reached after applying the action
        and the environment has not terminated the episode, `step()` should
        return a `EnvStep` with `step_type==StepType.TIMEOUT`.

        If possible, update the visualization display as well.

        Args:
            action (object): A NumPy array, or a nested dict, list or tuple
                of arrays conforming to `action_space`.

        Returns:
            EnvStep: The time step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.

        """

    @abc.abstractmethod
    def render(self, mode):
        """Renders the environment.

        The set of supported modes varies per environment. By convention,
        if mode is:

        * rgb_array: Return an `numpy.ndarray` with shape (x, y, 3) and type
            uint8, representing RGB values for an x-by-y pixel image, suitable
            for turning into a video.
        * ansi: Return a string (str) or `StringIO.StringIO` containing a
            terminal-style text representation. The text can include newlines
            and ANSI escape sequences (e.g. for colors).

        Make sure that your class's `render_modes` includes the list of
        supported modes.

        For example:

        .. code-block:: python

            class MyEnv(Environment):
                def render_modes(self):
                    return ['rgb_array', 'ansi']

                def render(self, mode):
                    if mode == 'rgb_array':
                        return np.array(...)  # return RGB frame for video
                    elif mode == 'ansi':
                        ...  # return text output
                    else:
                        raise ValueError('Supported render modes are {}, but '
                                         'got render mode {} instead.'.format(
                                             self.render_modes, mode))

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        """

    @abc.abstractmethod
    def visualize(self):
        """Creates a visualization of the environment.

        This function should be called **only once** after `reset()` to set up
        the visualization display. The visualization should be updated
        when the environment is changed (i.e. when `step()` is called.)

        Calling `close()` will deallocate any resources and close any
        windows created by `visualize()`. If `close()` is not explicitly
        called, the visualization will be closed when the environment is
        destructed (i.e. garbage collected).

        """

    @abc.abstractmethod
    def close(self):
        """Closes the environment.

        This method should close all windows invoked by `visualize()`.

        Override this function in your subclass to perform any necessary
        cleanup.

        Environments will automatically `close()` themselves when they are
        garbage collected or when the program exits.
        """

    def _validate_render_mode(self, mode):
        if mode not in self.render_modes:
            raise ValueError('Supported render modes are {}, but '
                             'got render mode {} instead.'.format(
                                 self.render_modes, mode))

    def __del__(self):
        """Environment destructor."""
        self.close()
