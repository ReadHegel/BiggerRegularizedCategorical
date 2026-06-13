import os
import sys

ARCH = "xqc"

os.environ.setdefault("CUDA_ROOT", "/usr")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
from jaxrl.agent.brc_learner import BRC
from jaxrl.env_names import get_environment_list
from jaxrl.envs import ParallelEnv

env = ParallelEnv(get_environment_list("DMC_DOGS"), seed=0)
agent = BRC(
    0,
    env.observation_space.sample()[:1],
    env.action_space.sample()[:1],
    len(env.envs),
    arch=ARCH,
    updates_per_step=2,
)
n = sum(p.size for p in jax.tree_util.tree_leaves(agent.critic.params))
print(f"arch={ARCH} critic.params: {n:,} ({n / 1e6:.3f}M)")
