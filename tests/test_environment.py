import akro
import cloudpickle
import gym
import numpy as np
import pytest

from garage import EnvSpec, EnvStep, StepType


class TestEnvSpec:

    def test_pickleable(self):
        env_spec = EnvSpec(akro.Box(-1, 1, (1, )), akro.Box(-2, 2, (2, )), 500)
        round_trip = cloudpickle.loads(cloudpickle.dumps(env_spec))
        assert round_trip
        assert round_trip.action_space == env_spec.action_space
        assert round_trip.observation_space == env_spec.observation_space
        assert round_trip.max_episode_length == env_spec.max_episode_length


@pytest.fixture
def sample_data():
    # spaces
    obs_space = gym.spaces.Box(low=1,
                               high=10,
                               shape=(4, 3, 2),
                               dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)

    # generate data
    obs = obs_space.sample()
    next_obs = obs_space.sample()
    act = act_space.sample()
    rew = 10.0
    step_type = StepType.FIRST

    # env_infos
    env_infos = dict()
    env_infos['goal'] = np.array([[1, 1]])
    env_infos['TimeLimit.truncated'] = (step_type == StepType.TIMEOUT)

    # agent_infos
    agent_infos = dict()
    agent_infos['prev_action'] = act

    return {
        'env_spec': env_spec,
        'observation': obs,
        'next_observation': next_obs,
        'action': act,
        'reward': rew,
        'env_info': env_infos,
        'agent_info': agent_infos,
        'step_type': step_type
    }


def test_new_env_step(sample_data):
    del sample_data['agent_info']
    s = EnvStep(**sample_data)
    assert s.env_spec is sample_data['env_spec']
    assert s.observation is sample_data['observation']
    assert s.action is sample_data['action']
    assert s.reward is sample_data['reward']
    assert s.step_type is sample_data['step_type']
    assert s.env_info is sample_data['env_info']
    del s

    obs_space = akro.Box(low=-1, high=10, shape=(4, 3, 2), dtype=np.float32)
    act_space = akro.Box(low=-1, high=10, shape=(4, 2), dtype=np.float32)
    env_spec = EnvSpec(obs_space, act_space)
    sample_data['env_spec'] = env_spec
    obs_space = akro.Box(low=-1000,
                         high=1000,
                         shape=(4, 3, 2),
                         dtype=np.float32)
    act_space = akro.Box(low=-1000, high=1000, shape=(4, 2), dtype=np.float32)
    sample_data['observation'] = obs_space.sample()
    sample_data['next_observation'] = obs_space.sample()
    sample_data['action'] = act_space.sample()
    s = EnvStep(**sample_data)

    assert s.observation is sample_data['observation']
    assert s.next_observation is sample_data['next_observation']
    assert s.action is sample_data['action']


def test_obs_env_spec_mismatch_env_step(sample_data):
    del sample_data['agent_info']
    with pytest.raises(ValueError,
                       match='observation must conform to observation_space'):
        sample_data['observation'] = sample_data['observation'][:, :, :1]
        s = EnvStep(**sample_data)
        del s

    obs_space = akro.Box(low=1, high=10, shape=(4, 5, 2), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)
    sample_data['env_spec'] = env_spec

    with pytest.raises(
            ValueError,
            match='observation should have the same dimensionality'):
        sample_data['observation'] = sample_data['observation'][:, :, :1]
        s = EnvStep(**sample_data)
        del s


def test_next_obs_env_spec_mismatch_env_step(sample_data):
    del sample_data['agent_info']
    with pytest.raises(
            ValueError,
            match='next_observation must conform to observation_space'):
        sample_data['next_observation'] = sample_data[
            'next_observation'][:, :, :1]
        s = EnvStep(**sample_data)
        del s

    obs_space = akro.Box(low=1, high=10, shape=(4, 3, 2), dtype=np.float32)
    act_space = gym.spaces.MultiDiscrete([2, 5])
    env_spec = EnvSpec(obs_space, act_space)
    sample_data['env_spec'] = env_spec

    with pytest.raises(
            ValueError,
            match='next_observation should have the same dimensionality'):
        sample_data['next_observation'] = sample_data[
            'next_observation'][:, :, :1]
        s = EnvStep(**sample_data)
        del s


def test_act_env_spec_mismatch_env_step(sample_data):
    del sample_data['agent_info']
    with pytest.raises(ValueError,
                       match='action must conform to action_space'):
        sample_data['action'] = sample_data['action'][:-1]
        s = EnvStep(**sample_data)
        del s

    obs_space = akro.Box(low=1, high=10, shape=(4, 3, 2), dtype=np.float32)
    act_space = akro.Discrete(5)
    env_spec = EnvSpec(obs_space, act_space)
    sample_data['env_spec'] = env_spec

    with pytest.raises(ValueError,
                       match='action should have the same dimensionality'):
        sample_data['action'] = sample_data['action'][:-1]
        s = EnvStep(**sample_data)
        del s


def test_reward_dtype_mismatch_env_step(sample_data):
    del sample_data['agent_info']
    with pytest.raises(ValueError, match='reward must be type'):
        sample_data['reward'] = []
        s = EnvStep(**sample_data)
        del s


def test_env_info_dtype_mismatch_env_step(sample_data):
    del sample_data['agent_info']
    with pytest.raises(ValueError, match='env_info must be type'):
        sample_data['env_info'] = []
        s = EnvStep(**sample_data)
        del s


def test_step_type_dtype_mismatch_env_step(sample_data):
    del sample_data['agent_info']
    with pytest.raises(ValueError, match='step_type must be dtype'):
        sample_data['step_type'] = []
        s = EnvStep(**sample_data)
        del s


def test_step_type_property_env_step(sample_data):
    del sample_data['agent_info']
    sample_data['step_type'] = StepType.FIRST
    s = EnvStep(**sample_data)
    assert s.first

    sample_data['step_type'] = StepType.MID
    s = EnvStep(**sample_data)
    assert s.mid

    sample_data['step_type'] = StepType.TERMINAL
    s = EnvStep(**sample_data)
    assert s.terminal and s.last

    sample_data['step_type'] = StepType.TIMEOUT
    s = EnvStep(**sample_data)
    assert s.timeout and s.last
