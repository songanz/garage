import gym
import pytest

from garage import EnvSpec
from garage.envs import GymEnv
from garage.envs.bullet import BulletEnv


class TestGymEnv:

    def test_wraps_env_spec(self):
        garage_env = GymEnv(env='Pendulum-v0')
        assert isinstance(garage_env.spec, EnvSpec)

    def test_closes_box2d(self):
        garage_env = GymEnv(env='CarRacing-v0')
        garage_env.visualize()
        assert garage_env._env.viewer is not None
        garage_env.close()
        assert garage_env._env.viewer is None

    @pytest.mark.mujoco
    def test_closes_mujoco(self):
        garage_env = GymEnv(env='Ant-v2')
        garage_env.visualize()
        assert garage_env._env.viewer is not None
        garage_env.close()
        assert garage_env._env.viewer is None

    def test_time_limit_env(self):
        garage_env = GymEnv(env='Pendulum-v0', max_episode_length=200)
        garage_env._env._max_episode_steps = 200
        garage_env.reset()
        for _ in range(200):
            es = garage_env.step(garage_env.spec.action_space.sample())
        assert es.timeout and es.env_info['TimeLimit.truncated']
        assert es.env_info['GymEnv.TimeLimitTerminated']

    def test_process_env_argument(self):
        env = GymEnv(env=gym.make('Ant-v2'))
        env = GymEnv(env='Ant-v2')
        env = GymEnv(gym.make('Ant-v2'))
        env = GymEnv('Ant-v2')
        with pytest.raises(ValueError, match='GymEnv should can take env'):
            env = GymEnv(1)

    def test_return_bullet_env(self):
        env = GymEnv(env=gym.make('CartPoleBulletEnv-v1'))
        assert isinstance(env, BulletEnv)
        env = GymEnv(env='CartPoleBulletEnv-v1')
        assert isinstance(env, BulletEnv)
        env = GymEnv(gym.make('CartPoleBulletEnv-v1'))
        assert isinstance(env, BulletEnv)
        env = GymEnv('CartPoleBulletEnv-v1')
        assert isinstance(env, BulletEnv)
