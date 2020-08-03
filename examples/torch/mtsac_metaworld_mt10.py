#!/usr/bin/env python3
"""MTSAC implementation based on Metaworld. Benchmarked on MT10.

https://arxiv.org/pdf/1910.10897.pdf
"""
import click
import metaworld
import numpy as np
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GarageEnv, TaskOnehotWrapper, normalize
from garage.experiment import deterministic, LocalRunner
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


@click.command()
@click.option('--seed', 'seed', type=int, default=1)
@click.option('--gpu', '_gpu', type=int, default=None)
@wrap_experiment(snapshot_mode='none')
def mtsac_metaworld_mt10(ctxt=None, seed=1, _gpu=None):
    """Train MTSAC with MT10 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).

    """
    deterministic.set_seed(seed)
    runner = LocalRunner(ctxt)
    mt10 = metaworld.MT10()
    mt10_test = metaworld.MT10()
    task_indices = {env_name: index
                    for (index, env_name)
                    in enumerate(mt10.train_classes.keys())}

    def wrap(env, task):
        normalized = normalize(GarageEnv(env), scale_reward=True)
        with_onehot = TaskOnehotWrapper(normalized,
                                        task_index=task_indices[task.env_name],
                                        n_total_tasks=len(task_indices))
        return with_onehot

    train_task_sampler = MetaWorldTaskSampler(mt10, 'train', wrap)
    test_task_sampler = MetaWorldTaskSampler(mt10_test, 'train', wrap)

    mt10_train_envs = train_task_sampler.sample(10)
    mt10_test_envs = test_task_sampler.sample(10)

    policy = TanhGaussianMLPPolicy(
        env_spec=mt10_train_envs.spec,
        hidden_sizes=[400, 400, 400],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=mt10_train_envs.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=mt10_train_envs.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )

    timesteps = 20000000
    batch_size = int(150 * mt10_train_envs.num_tasks)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=150,
                  max_path_length=150,
                  eval_env=mt10_test_envs,
                  env_spec=mt10_train_envs.spec,
                  num_tasks=10,
                  steps_per_epoch=epoch_cycles,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1500,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=1280)
    if _gpu is not None:
        set_gpu_mode(True, _gpu)
    mtsac.to()
    runner.setup(algo=mtsac, env=mt10_train_envs, sampler_cls=LocalSampler)
    runner.train(n_epochs=epochs, batch_size=batch_size)


mtsac_metaworld_mt10()
