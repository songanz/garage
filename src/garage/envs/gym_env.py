"""Wrapper class that converts gym.Env into GymEnv."""

import copy
import math

import akro
import gym
from gym.wrappers.time_limit import TimeLimit

from garage import Environment, EnvSpec, EnvStep, StepType

# The gym environments using one of the packages in the following lists as
# entry points don't close their viewer windows.
KNOWN_GYM_NOT_CLOSE_VIEWER = [
    # Please keep alphabetized
    'gym.envs.atari',
    'gym.envs.box2d',
    'gym.envs.classic_control'
]

KNOWN_GYM_NOT_CLOSE_MJ_VIEWER = [
    # Please keep alphabetized
    'gym.envs.mujoco',
    'gym.envs.robotics'
]


def _get_env_time_limit(env):
    """Get time limit from a gym.Env.

    Args:
        env (gym.Env): the input gym.Env

    Returns:
        int: if there max_episode_length is found in the gym.Env. Or None if
        not found.

    Raises:
        RuntimeError: if the gym.Env is wrapped by a gym.TimeLimit,
        and env.spec._max_episode_steps and env._max_episode_steps don't match.
    """
    gym_spec_steps = None
    if hasattr(env, 'spec') and env.spec:
        gym_spec_steps = env.spec.max_episode_steps
    else:
        # metaworld env doesn't have spec, what to do?
        pass

    if isinstance(env, TimeLimit):
        # pylint: disable=protected-access
        timelimit_steps = env._max_episode_steps
        if gym_spec_steps and gym_spec_steps != timelimit_steps:
            raise RuntimeError('Expect wrapped gym environment '
                               ' TimeLimit._max_episode_steps '
                               'be equal to env.spec.max_episode_stpes'
                               '({}), but got {} instead'.format(
                                   timelimit_steps, gym_spec_steps))

    return gym_spec_steps


class GymEnv(Environment):
    """Returns an abstract Garage wrapper class for gym.Env.

    In order to provide pickling (serialization) and parameterization
    for gym.Envs, they must be wrapped with a GymEnv. This ensures
    compatibility with existing samplers and checkpointing when the
    envs are passed internally around garage.

    Furthermore, classes inheriting from GymEnv should silently
    convert action_space and observation_space from gym.Spaces to
    akro.spaces.

    GymEnv handles all environments created by gym.make().
    It returns a different wrapper class instance if the input environment
    requires special handling.
    Current supported wrapper classes are:
        garage.envs.bullet.BulletEnv for Bullet-based gym environments.
    See __new__() for details.
    """

    def __new__(cls, *args, **kwargs):
        """Returns environment specific wrapper based on input environment type.

        Args:
            args: positional arguments
            kwargs: keyword arguments

        Returns:
             garage.envs.bullet.BulletEnv: if the environment is a bullet-based
                environment. Else returns a garage.envs.GymEnv
        """
        # pylint: disable=import-outside-toplevel
        # Determine if the input env is a bullet-based gym environment
        env = None
        if 'env' in kwargs:  # env passed as a keyword arg
            env = kwargs['env']
        elif len(args) >= 1:
            # env passed as a positional arg
            env = args[0]

        if isinstance(env, gym.Env):
            if issubclass(env.__class__, gym.Wrapper):
                env = env.unwrapped

            if env.spec and env.spec.id.find('Bullet') >= 0:
                from garage.envs.bullet import BulletEnv
                return BulletEnv(*args, **kwargs)
        elif isinstance(env, str):
            if 'Bullet' in env:
                from garage.envs.bullet import BulletEnv
                return BulletEnv(*args, **kwargs)

        return super(GymEnv, cls).__new__(cls)

    def __init__(self, env, is_image=False, max_episode_length=None):
        """Initializes a GymEnv.

        Note that if `env` and `env_name` are passed in at the same time,
        `env` will be wrapped.

        Args:
            env (gym.wrappers.time_limit or str): A gym.TimeLimit
                object wrapping a gym.Env created via gym.make(). Or a name
                of the gym environment to be created.
            is_image (bool): True if observations contain pixel values,
                false otherwise. Setting this to true converts a gym.Spaces.Box
                obs space to an akro.Image and normalizes pixel values.
            max_episode_length (int): The maximum steps allowed for an episode.

        Raises:
            ValueError: if `env` neither a gym.Env object nor a string.
            RuntimeError: if the environment is wrapped by a TimeLimit and its
                max_episode_steps is not equal to its spec's time limit value.
        """
        if isinstance(env, str):
            self._env = gym.make(env)
        elif isinstance(env, gym.Env):
            self._env = env
        else:
            raise ValueError('GymEnv should can take env as either a string, '
                             'or an Gym environment, but got type {} '
                             'instead.'.format(type(env)))

        env_time_limit = _get_env_time_limit(self._env)
        if max_episode_length and env_time_limit:
            if max_episode_length != env_time_limit:
                raise RuntimeError('Expect max_episode_length to '
                                   'be equal to '
                                   'env.spec.max_episode_stpes'
                                   '({}), but got {} instead'.format(
                                       env_time_limit, max_episode_length))
        elif not max_episode_length and env_time_limit:
            max_episode_length = env_time_limit
        elif not max_episode_length and not env_time_limit:
            max_episode_length = math.inf
        else:
            # if max_episode_length and not env_time_limit,
            # use max_episode_length
            pass
        self._max_episode_length = max_episode_length

        self._render_modes = self._env.metadata['render.modes']

        self._last_observation = None
        self._step_cnt = 0
        self._visualize = False

        self._action_space = akro.from_gym(self._env.action_space)
        self._observation_space = akro.from_gym(self._env.observation_space,
                                                is_image=is_image)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """garage.envs.env_spec.EnvSpec: The envionrment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return self._render_modes

    def reset(self):
        """Call reset on wrapped env.

        Returns:
            numpy.ndarray: The first observation. It must conform to
                `observation_space`.
            dict: The episode-level information. Note that this is not part
                of `env_info` provided in `step()`. It contains information of
                the entire episode， which could be needed to determine the
                first action (e.g. in the case of goal-conditioned or MTRL.)

        """
        first_obs = self._env.reset()

        self._step_cnt = 0
        self._last_observation = first_obs

        return first_obs, dict()

    def step(self, action):
        """Call step on wrapped env.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            EnvStep: The time step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.

        """
        if self._last_observation is None:
            raise RuntimeError('reset() must be called before step()!')

        observation, reward, done, info = self._env.step(action)

        if self._visualize:
            self._env.render(mode='human')

        last_obs = self._last_observation
        # Type conversion
        if not isinstance(reward, float):
            reward = float(reward)

        self._last_observation = observation
        self._step_cnt += 1

        step_type = None
        if done:
            step_type = StepType.TERMINAL
        elif self._step_cnt == 1:
            step_type = StepType.FIRST
        else:
            step_type = StepType.MID

        # gym envs that are wrapped in TimeLimit wrapper modify
        # the done/termination signal to be true whenever a time
        # limit expiration occurs. The following statement sets
        # the done signal to be True only if caused by an
        # environment termination, and not a time limit
        # termination. The time limit termination signal
        # will be saved inside env_infos as
        # 'GymEnv.TimeLimitTerminated'
        if (self._step_cnt >= self._spec.max_episode_length
                or 'TimeLimit.truncated' in info):
            info['GymEnv.TimeLimitTerminated'] = True
            step_type = StepType.TIMEOUT
        else:
            info['TimeLimit.truncated'] = False
            info['GymEnv.TimeLimitTerminated'] = False

        return EnvStep(env_spec=self.spec,
                       observation=last_obs,
                       action=action,
                       reward=reward,
                       next_observation=observation,
                       env_info=info,
                       step_type=step_type)

    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        Returns:
            object: the return value for render, depending on each
                environment's implementation.
        """
        self._validate_render_mode(mode)
        return self._env.render(mode)

    def visualize(self):
        """Creates a visualization of the environment."""
        self._env.render(mode='human')
        self._visualize = True

    def close(self):
        """Close the wrapped env."""
        self._close_viewer_window()
        self._env.close()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Call wrapped environment's compute_reward function.

        Args:
            achieved_goal (object): current achieved_goal.
            desired_goal (object): desired goal.
            info (info): info to take.

        Returns:
            object: the reward.
        """
        return self._env.compute_reward(achieved_goal, desired_goal, info)

    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, float]]: A list of "tasks," where each task is a
                dictionary containing a single key, "direction", mapping to -1
                or 1.

        """
        return self._env.sample_tasks(num_tasks)

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, float]): A task (a dictionary containing a single
                key, "direction", mapping to -1 or 1).

        """
        self._env.set_task(task)

    def _close_viewer_window(self):
        """Close viewer window.

        Unfortunately, some gym environments don't close the viewer windows
        properly, which leads to "out of memory" issues when several of
        these environments are tested one after the other.
        This method searches for the viewer object of type MjViewer, Viewer
        or SimpleImageViewer, based on environment, and if the environment
        is wrapped in other environment classes, it performs depth search
        in those as well.
        This method can be removed once OpenAI solves the issue.
        """
        # We need to do some strange things here to fix-up flaws in gym
        # pylint: disable=import-outside-toplevel
        if self._env.spec:
            if any(package in getattr(self._env.spec, 'entry_point', '')
                   for package in KNOWN_GYM_NOT_CLOSE_MJ_VIEWER):
                # This import is not in the header to avoid a MuJoCo dependency
                # with non-MuJoCo environments that use this base class.
                try:
                    from mujoco_py.mjviewer import MjViewer
                    import glfw
                except ImportError:
                    # If we can't import mujoco_py, we must not have an
                    # instance of a class that we know how to close here.
                    return
                if (hasattr(self._env, 'viewer')
                        and isinstance(self._env.viewer, MjViewer)):
                    glfw.destroy_window(self._env.viewer.window)
            elif any(package in getattr(self._env.spec, 'entry_point', '')
                     for package in KNOWN_GYM_NOT_CLOSE_VIEWER):
                if hasattr(self._env, 'viewer'):
                    from gym.envs.classic_control.rendering import (
                        Viewer, SimpleImageViewer)
                    if (isinstance(self._env.viewer,
                                   (SimpleImageViewer, Viewer))):
                        self._env.viewer.close()

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s dictionary to be pickled.

        """
        # the viewer object is not pickleable
        # we first make a copy of the viewer
        env = self._env

        # get the inner env if it is a gym.Wrapper
        if issubclass(env.__class__, gym.Wrapper):
            env = env.unwrapped

        if 'viewer' in env.__dict__:
            _viewer = env.viewer
            # remove the viewer and make a copy of the state
            env.viewer = None
            state = copy.deepcopy(self.__dict__)
            # assign the viewer back to self.__dict__
            env.viewer = _viewer
            # the returned state doesn't have the viewer
            return state
        return self.__dict__

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(state['_env'])
