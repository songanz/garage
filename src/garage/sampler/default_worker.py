"""Default Worker class."""
from collections import defaultdict

import numpy as np

from garage import StepType, TrajectoryBatch
from garage.experiment import deterministic
from garage.sampler import _apply_env_update
from garage.sampler.worker import Worker


class DefaultWorker(Worker):
    """Initialize a worker.

    Args:
        seed(int): The seed to use to intialize random number generators.
        max_episode_length(int or float): The maximum length paths which will
            be sampled. Can be (floating point) infinity.
        worker_number(int): The number of the worker where this update is
            occurring. This argument is used to set a different seed for each
            worker.

    Attributes:
        agent(Policy or None): The worker's agent.
        env(gym.Env or None): The worker's environment.

    """

    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_episode_length,
            worker_number):
        super().__init__(seed=seed,
                         max_episode_length=max_episode_length,
                         worker_number=worker_number)
        self.agent = None
        self.env = None
        self._observations = []
        self._last_observations = []
        self._actions = []
        self._rewards = []
        self._step_types = []
        self._lengths = []
        self._agent_infos = defaultdict(list)
        self._env_infos = defaultdict(list)
        self._prev_obs = None
        self._path_length = 0
        self.worker_init()

    def worker_init(self):
        """Initialize a worker."""
        if self._seed is not None:
            deterministic.set_seed(self._seed + self._worker_number)

    def update_agent(self, agent_update):
        """Update an agent, assuming it implements garage.Policy.

        Args:
            agent_update (np.ndarray or dict or garage.Policy): If a
                tuple, dict, or np.ndarray, these should be parameters to
                agent, which should have been generated by calling
                `policy.get_param_values`. Alternatively, a policy itself. Note
                that other implementations of `Worker` may take different types
                for this parameter.

        """
        if isinstance(agent_update, (dict, tuple, np.ndarray)):
            self.agent.set_param_values(agent_update)
        elif agent_update is not None:
            self.agent = agent_update

    def update_env(self, env_update):
        """Use any non-None env_update as a new environment.

        A simple env update function. If env_update is not None, it should be
        the complete new environment.

        This allows changing environments by passing the new environment as
        `env_update` into `obtain_samples`.

        Args:
            env_update(gym.Env or EnvUpdate or None): The environment to
                replace the existing env with. Note that other implementations
                of `Worker` may take different types for this parameter.

        Raises:
            TypeError: If env_update is not one of the documented types.

        """
        self.env, _ = _apply_env_update(self.env, env_update)

    def start_rollout(self):
        """Begin a new rollout."""
        self._path_length = 0
        self._prev_obs, _ = self.env.reset()
        self.agent.reset()

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination of due to reaching `max_episode_length`.

        """
        if self._path_length < self._max_episode_length:
            a, agent_info = self.agent.get_action(self._prev_obs)
            es = self.env.step(a)
            self._observations.append(self._prev_obs)
            self._rewards.append(es.reward)
            self._actions.append(es.action)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k, v in es.env_info.items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._step_types.append(es.step_type)
            if not es.last:
                self._prev_obs = es.next_observation
                return False
        self._lengths.append(self._path_length)
        self._last_observations.append(self._prev_obs)
        return True

    def collect_rollout(self):
        """Collect the current rollout, clearing the internal buffer.

        Returns:
            garage.TrajectoryBatch: A batch of the trajectories completed since
                the last call to collect_rollout().

        """
        observations = self._observations
        self._observations = []
        last_observations = self._last_observations
        self._last_observations = []
        actions = self._actions
        self._actions = []
        rewards = self._rewards
        self._rewards = []
        step_types = self._step_types
        self._step_types = []
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        lengths = self._lengths
        self._lengths = []
        return TrajectoryBatch(env_spec=self.env.spec,
                               observations=np.asarray(observations),
                               last_observations=np.asarray(last_observations),
                               actions=np.asarray(actions),
                               rewards=np.asarray(rewards),
                               step_types=np.asarray(step_types,
                                                     dtype=StepType),
                               env_infos=dict(env_infos),
                               agent_infos=dict(agent_infos),
                               lengths=np.asarray(lengths, dtype='i'))

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.start_rollout()
        while not self.step_rollout():
            pass
        return self.collect_rollout()

    def shutdown(self):
        """Close the worker's environment."""
        self.env.close()
