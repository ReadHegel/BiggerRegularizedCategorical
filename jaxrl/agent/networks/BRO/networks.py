from __future__ import annotations

from typing import Callable

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl.agent.networks.common import BaseActor, BaseEnsembleMultitaskCritic

BRO_ACTOR_CONFIG = {
    "hidden_dims": 256,
    "depth": 1,
    "log_std_scale": 1.0,
    "log_std_min": -10.0,
    "log_std_max": 2.0,
}
BRO_CRITIC_CONFIG = {
    "hidden_dims": 512,
    "depth": 2,
}


def make_bro_actor(action_dim: int, **overrides) -> NormalTanhPolicy:
    cfg = {**BRO_ACTOR_CONFIG, **overrides, "action_dim": action_dim}
    return NormalTanhPolicy(**cfg)


def make_bro_critic(
    *,
    num_tasks: int,
    embedding_size: int,
    ensemble_size: int,
    num_bins: int,
    multitask: bool,
    **overrides,
) -> Critic:
    cfg = {
        **BRO_CRITIC_CONFIG,
        **overrides,
        "num_tasks": num_tasks,
        "embedding_size": embedding_size,
        "ensemble_size": ensemble_size,
        "num_bins": num_bins,
        "multitask": multitask,
    }
    return Critic(**cfg)


def default_init(scale: float = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class BronetBlock(nn.Module):
    hidden_dims: int
    activations: Callable[[jnp.ndarray], jnp.ndarray]

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        res = nn.Dense(self.hidden_dims, kernel_init=default_init())(x)
        res = nn.LayerNorm()(res)
        res = self.activations(res)
        res = nn.Dense(self.hidden_dims, kernel_init=default_init())(res)
        res = nn.LayerNorm()(res)
        return res + x


class BroNet(nn.Module):
    hidden_dims: int
    depth: int
    add_final_layer: bool = False
    output_nodes: int = 101
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.hidden_dims, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = self.activations(x)
        for _ in range(self.depth):
            x = BronetBlock(self.hidden_dims, self.activations)(x)
        if self.add_final_layer:
            x = nn.Dense(self.output_nodes, kernel_init=default_init())(x)
        return x


class QValue(nn.Module):
    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_nodes: int = 101

    def setup(self):
        self.critic = BroNet(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            activations=self.activations,
            add_final_layer=True,
            output_nodes=self.output_nodes,
        )

    def __call__(self, inputs: jnp.ndarray, training):
        return self.critic(inputs)


class NormalTanhPolicy(BaseActor):
    action_dim: int
    hidden_dims: int = 256
    depth: int = 1
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    log_std_scale: float = 1.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray, temperature: float = 1.0, training: bool = False):
        outputs = BroNet(
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            activations=self.activations,
            add_final_layer=False,
            output_nodes=None,
        )(observations)
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(self.log_std_scale))(outputs)
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
            1 + nn.tanh(log_stds)
        )
        stds = jnp.exp(log_stds) * temperature
        base_dist = distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)
        return distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), 1))


class Critic(BaseEnsembleMultitaskCritic):
    q_module = QValue

    hidden_dims: int = 512
    depth: int = 2
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_bins: int = 101

    def q_member_kwargs(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "depth": self.depth,
            "activations": self.activations,
            "output_nodes": self.num_bins,
        }
