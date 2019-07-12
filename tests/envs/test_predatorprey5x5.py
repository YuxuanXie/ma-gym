import gym
import pytest
import ma_gym


@pytest.fixture(scope='module')
def env():
    env = gym.make('PredatorPrey5x5-v0')
    yield env
    env.close()


def test_init(env):
    assert env.n_agents == 2


def test_reset(env):
    env.reset()
    assert env._step_count == 0
    assert env._total_episode_reward == 0
    assert env._agent_dones == [False for _ in range(env.n_agents)]
    assert env._prey_alive == [True for _ in range(env.n_preys)]