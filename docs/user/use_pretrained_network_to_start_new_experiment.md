# Use a pre-trained network to start a new experiment

In this section you will learn how to load a pre-trained network and use it in
new experiments. We'll cover two cases in particular:

- How to use a trained policy as an expert in Behavioral Cloning
- How to reuse a trained Q function in DQN

Before attempting either of these, you'll need a saved experiment snapshot. [This page](https://garage.readthedocs.io/en/latest/user/save_load_resume_exp.html)
will show you how to get one.

## Using a pre-trained policy as a BC expert

There are two steps involved. First, we must load the pre-trained policy. Assuming
that it was trained with garage, details on extracting a policy from a saved experiment
can be found [here](https://garage.readthedocs.io/en/latest/user/reuse_garage_policy.html).
Next, we setup a new experiment and pass the policy as the `source` argument of
the `BC` constructor:

```python
# Load the policy
from garage.experiment import Snapshotter
snapshotter = Snapshotter()
snapshot = snapshotter.load('path/to/snapshot/dir')

expert = snapshot['algo'].policy
env = snapshot['env']  # We assume env is the same

# Setup new experiment
from garage import wrap_experiment
from garage.experiment import LocalRunner
from garage.torch.algos import BC
from garage.torch.policies import GaussianMLPPolicy

@wrap_experiment
def bc_with_pretrained_expert(ctxt=None):
    runner = LocalRunner(ctxt)
    policy = GaussianMLPPolicy(env.spec, [8, 8])
    batch_size = 1000
    algo = BC(env.spec,
              policy,
              batch_size=batch_size,
              source=expert,
              max_path_length=200,
              policy_lr=1e-2,
              loss='log_prob')
    runner.setup(algo, env)
    runner.train(100, batch_size=batch_size)


bc_with_pretrained_expert()
```

Please refer to [this page](https://garage.readthedocs.io/en/latest/user/algo_bc.html)
for more information on garage's implementation of Behavioral Cloning. If your expert
policy wasn't trained with garage, you can wrap it in garage's `Policy` API
(`garage.torch.policies.Policy`) before passing it to `BC`.

## Using a pre-trained Q function in a new DQN experiment

Garage's DQN module accepts a Q function in its constructor: `DQN(env_space=env.spec, policy=policy, qf=qf, ...)`
To use a pre-trained Q function, we simply load one and pass it in, rather than
creating a new one. Since there is a relatively large number of constructs that
go into creating a DQN, we suggest you use the [Pong example code](https://github.com/rlworkgroup/garage/blob/master/examples/tf/dqn_pong.py)
as a starting point. Make the following modifications to reuse a Q function:

```python
# At the top of the file, add:
from garage.experiment import Snapshotter

# Replace `qf = DiscreteCNNQFunction(...)` with:
snapshotter = Snapshotter()
snapshot = snapshotter.load('path/to/previous/run/snapshot/dir')
qf = snapshot['algo']._qf
```

----

_This page was authored by Hayden Shively
([@haydenshively](https://github.com/haydenshively))_
