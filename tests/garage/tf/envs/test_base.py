import pickle

import gym
import pytest

from garage.envs import GymEnv
from garage.envs.bullet import _get_unsupported_env_list

from tests.helpers import step_env_with_gym_quirks


class TestGymEnv:

    def test_is_pickleable(self):
        env = GymEnv(env='CartPole-v1')
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs(self, spec):
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        if any(name == spec.id for name in _get_unsupported_env_list()):
            pytest.skip('Skip unsupported Bullet environments')
        env = GymEnv(env=spec.id)
        step_env_with_gym_quirks(env, spec)

    @pytest.mark.nightly
    @pytest.mark.parametrize('spec', list(gym.envs.registry.all()))
    def test_all_gym_envs_pickleable(self, spec):
        if spec._env_name.startswith('Defender'):
            pytest.skip(
                'Defender-* envs bundled in atari-py 0.2.x don\'t load')
        if any(name == spec.id for name in _get_unsupported_env_list()):
            pytest.skip('Skip unsupported Bullet environments')
        env = GymEnv(env=spec.id)
        step_env_with_gym_quirks(env,
                                 spec,
                                 n=1,
                                 visualize=True,
                                 serialize_env=True)
