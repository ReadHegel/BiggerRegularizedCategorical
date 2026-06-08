"""Implementations of algorithms for continuous control."""

from __future__ import annotations

from typing import Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
import distrax

from jaxrl.agent.networks.common import BaseActor, BaseEnsembleMultitaskCritic
from jaxrl.agent.networks.XQC.mlp import (
    MLP,
    BatchNormEmbedder,
    CrossQBlock,
    DenseBlock,
    IdentityEmbedder,
    LayerNormEmbedder,
    LNBlock,
    ScalarPredictor,
    TanhGaussPredictor,
    XQCBlock,
)
from jaxrl.agent.networks.XQC.utils import norm_network
from jaxrl.utils import Model


XQC_ACTOR_CONFIG = {
    "hidden_dims": (256, 256, 256, 256),
    "pre_activation_bn": True,
    "use_layer_norm": False,
    "use_batch_norm": True,
    "skip_connections": False,
}
XQC_CRITIC_CONFIG = {
    "hidden_dims": (228, 228, 228, 228),
    "pre_activation_bn": True,
    "use_layer_norm": False,
    "use_batch_norm": True,
    "skip_connections": False,
}


def make_xqc_actor(action_dim: int, **overrides) -> NormalTanhPolicy:
    cfg = {**XQC_ACTOR_CONFIG, **overrides, "action_dim": action_dim}
    return NormalTanhPolicy(**cfg)


def make_xqc_critic(
    *,
    num_tasks: int,
    embedding_size: int,
    ensemble_size: int,
    num_bins: int,
    multitask: bool,
    **overrides,
) -> Critic:
    cfg = {
        **XQC_CRITIC_CONFIG,
        **overrides,
        "num_tasks": num_tasks,
        "embedding_size": embedding_size,
        "ensemble_size": ensemble_size,
        "num_bins": num_bins,
        "multitask": multitask,
    }
    return Critic(**cfg)


class NormalTanhPolicy(BaseActor):
    hidden_dims: Sequence[int]
    action_dim: int
    pre_activation_bn: bool
    use_layer_norm: bool
    use_batch_norm: bool
    skip_connections: bool

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False,
    ) -> distrax.Distribution:
    
        if self.use_batch_norm:
            embedder = BatchNormEmbedder()
            block_class = XQCBlock if self.pre_activation_bn else CrossQBlock
        elif self.use_layer_norm:
            embedder = LayerNormEmbedder()
            block_class = LNBlock
        else:
            embedder = None
            block_class = DenseBlock

        mlp = MLP(
            embedder=embedder,
            predictor=TanhGaussPredictor(
                action_dim=self.action_dim, 
                temperature=temperature,
                name='predictor_tanh_gauss'
            ),
            hidden_dims=self.hidden_dims,
            block_class=block_class,
            skip_connections=self.skip_connections,
        )

        return mlp(observations, training=training)

    def post_update(self, model: Model) -> Model:
        return norm_network(model)


# @functools.partial(jax.jit, static_argnames=("actor_def", "temperature"))
# @functools.partial(jax.vmap, in_axes=(0, None, 0, 0, 0, None))
# def sample_actions_with_log_probs(
#     rng: PRNGKey,
#     actor_def: nn.Module,
#     actor_params: Params,
#     actor_batch_stats: Params,
#     observations: np.ndarray,
#     temperature: float = 1.0,
# ) -> Tuple[PRNGKey, jnp.ndarray]:
#     variables = {"params": actor_params}
#     if actor_batch_stats is not None:
#         variables["batch_stats"] = actor_batch_stats
#     dist = actor_def.apply(
#         variables,
#         observations,
#         temperature,
#     )
#     rng, key = jax.random.split(rng)
#     action = dist.sample(seed=key)
#     log_probs = dist.log_prob(action)
#     return rng, action, log_probs


class QValue(nn.Module):
    hidden_dims: Sequence[int]
    n_outputs: int
    pre_activation_bn: bool
    use_layer_norm: bool
    use_batch_norm: bool
    skip_connections: bool

    @nn.compact
    def __call__(
        self, inputs: jnp.ndarray, training: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = inputs

        # Determine Embedder
        if self.use_batch_norm:
            embedder = BatchNormEmbedder()
            block_class = XQCBlock if self.pre_activation_bn else CrossQBlock
        elif self.use_layer_norm:
            embedder = LayerNormEmbedder()
            block_class = LNBlock
        else:
            embedder = IdentityEmbedder()
            block_class = DenseBlock

        mlp = MLP(
            embedder=embedder,
            predictor=ScalarPredictor(n_outputs=self.n_outputs, name='predictor_scalar'),
            hidden_dims=self.hidden_dims,
            block_class=block_class,
            skip_connections=self.skip_connections,
        )
        values = mlp(x, training=training)

        # Scalar critic for MSE loss
        if self.n_outputs == 1:
            return jnp.squeeze(values, -1), None
        
        # Distributional critic
        else:
            return values # return logits
            # bin_values = jnp.linspace(
            #     self.min_v, 
            #     self.max_v, 
            #     values.shape[1], dtype=jnp.float32
            # )
            # log_probs = nn.log_softmax(values, axis=1)
            # values = jnp.sum(jnp.exp(log_probs) * bin_values, axis=1)
            # return values, log_probs

class Critic(BaseEnsembleMultitaskCritic):
    q_module = QValue

    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    pre_activation_bn: bool = True
    use_layer_norm: bool = False
    use_batch_norm: bool = True
    skip_connections: bool = False
    num_bins: int = 101

    def q_member_kwargs(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "n_outputs": self.num_bins,
            "pre_activation_bn": self.pre_activation_bn,
            "use_layer_norm": self.use_layer_norm,
            "use_batch_norm": self.use_batch_norm,
            "skip_connections": self.skip_connections,
        }

    def post_update(self, model: Model) -> Model:
        return norm_network(model)

# class VMapCritic(nn.Module):
#     max_v: float
#     min_v: float
#     hidden_dims: Sequence[int]
#     n_outputs: int
#     n_critics: int
#     pre_activation_bn: bool
#     use_layer_norm: bool
#     use_batch_norm: bool
#     skip_connections: bool

#     @nn.compact
#     def __call__(
#         self,
#         observations: jnp.ndarray,
#         actions: jnp.ndarray,
#         training: bool,
#         **kwargs,
#     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         q_values, log_probs = nn.vmap(
#             Critic,
#             variable_axes={"params": 0, "batch_stats": 0, "activations": 0},
#             split_rngs={"params": True, "batch_stats": True},
#             in_axes=None,
#             out_axes=0,
#             axis_size=self.n_critics,
#         )(  
#             self.hidden_dims,
#             n_outputs=self.n_outputs,
#             max_v=self.max_v,
#             min_v=self.min_v,
#             pre_activation_bn=self.pre_activation_bn,
#             use_layer_norm=self.use_layer_norm,
#             use_batch_norm=self.use_batch_norm,
#             skip_connections=self.skip_connections,
#         )(observations, actions, training)

#         return q_values, {"log_probs": log_probs}
