import math
from typing import Callable

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl.agent.networks.common import BaseActor, BaseEnsembleMultitaskCritic
from jaxrl.agent.networks.FlashSAC.flashsac_layer import (
    FlashSACBlock,
    CategoricalValue,
    FlashSACEmbedder,
    NormalTanhPolicy,
    UnitRMSNorm,
)
from jaxrl.agent.networks.SimbaV2.utils import tree_map_until_match
from jaxrl.utils import Model

# Default configs from the paper
FLASH_ACTOR_CONFIG = {
    "num_blocks": 2,
    "hidden_dim": 128,
}

FLASH_CRITIC_CONFIG = {
    "num_blocks": 2,
    "hidden_dim": 112,
}


def make_flashsac_actor(action_dim: int, **overrides) -> "FlashSACActor":
    cfg = {**FLASH_ACTOR_CONFIG, **overrides, "action_dim": action_dim}
    return FlashSACActor(**cfg)


def make_flashsac_critic(
    *,
    num_tasks: int,
    embedding_size: int,
    ensemble_size: int,
    num_bins: int,
    multitask: bool,
    **overrides,
) -> "FlashSACCritic":
    cfg = {
        **FLASH_CRITIC_CONFIG,
        **overrides,
        "num_tasks": num_tasks,
        "embedding_size": embedding_size,
        "ensemble_size": ensemble_size,
        "num_bins": num_bins,
        "multitask": multitask,
    }
    return FlashSACCritic(**cfg)


def l2normalize_flashsac_network(model: Model) -> Model:
    params = model.params

    # 1. Normalize all dense layers
    def norm_dense(tree):
        kernel = tree["kernel"]
        if len(kernel.shape) == 2:
            axis = 0
        elif len(kernel.shape) == 3: # For critic ensemble
            axis = 1
        l2norm = jnp.linalg.norm(kernel, ord=2, axis=axis, keepdims=True)
        new_kernel = kernel / jnp.maximum(l2norm, 1e-8)
        return {**tree, "kernel": new_kernel}

    params = tree_map_until_match(
        f=norm_dense, tree=params, target_re=".*unit_linear.*", keep_values=True
    )

    # 2. Normalize UnitBatchNorm scales and biases
    def norm_bn(tree):
        scale, bias = tree["weight"], tree["bias"]
        ndim = scale.shape[-1]
        sqsum = jnp.sum(scale**2 + bias**2, axis=-1, keepdims=True)
        norm_factor = math.sqrt(ndim) / jnp.sqrt(sqsum + 1e-8)
        return {
            **tree,
            "weight": scale * norm_factor,
            "bias": bias * norm_factor,
        }

    params = tree_map_until_match(
        f=norm_bn, tree=params, target_re=".*unit_batch_norm.*", keep_values=True
    )

    # 3. Normalize UnitRMSNorm scales
    def norm_rms(tree):
        scale = tree["weight"]
        ndim = scale.shape[-1]
        sqsum = jnp.sum(scale * scale, axis=-1, keepdims=True)
        norm_factor = math.sqrt(ndim) / jnp.sqrt(sqsum + 1e-8)
        return {
            **tree,
            "weight": scale * norm_factor,
        }

    params = tree_map_until_match(
        f=norm_rms, tree=params, target_re=".*(unit_rms_norm|post_norm).*", keep_values=True
    )

    return model.replace(params=params)


class FlashSACActor(BaseActor):
    num_blocks: int
    hidden_dim: int
    action_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = True,
    ) -> distrax.Distribution:
        # embedder
        x = FlashSACEmbedder(
            input_dim=observations.shape[-1],
            hidden_dim=self.hidden_dim,
            name="embedder",
        )(observations, training=training)

        # blocks
        for i in range(self.num_blocks):
            x = FlashSACBlock(
                hidden_dim=self.hidden_dim,
                name=f"encoder_{i}",
            )(x, training=training)

        x = UnitRMSNorm(name="post_norm")(x)

        # policy predictor
        dist = NormalTanhPolicy(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            name="predictor",
        )(x, temperature)
        
        return dist

    def post_update(self, model: Model) -> Model:
        return l2normalize_flashsac_network(model)


class FlashSACQValue(nn.Module):
    num_blocks: int
    hidden_dim: int
    num_bins: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # embedder
        x = FlashSACEmbedder(
            input_dim=inputs.shape[-1],
            hidden_dim=self.hidden_dim,
            name="embedder",
        )(inputs, training=training)

        # blocks
        for i in range(self.num_blocks):
            x = FlashSACBlock(
                hidden_dim=self.hidden_dim,
                name=f"encoder_{i}",
            )(x, training=training)

        x = UnitRMSNorm(name="post_norm")(x)

        # categorical predictor
        logits = CategoricalValue(
            hidden_dim=self.hidden_dim,
            num_bins=self.num_bins,
            name="predictor",
        )(x)
        
        return logits


class FlashSACCritic(BaseEnsembleMultitaskCritic):
    q_module = FlashSACQValue

    num_blocks: int = 2
    hidden_dim: int = 256
    num_bins: int = 101
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def q_member_kwargs(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "hidden_dim": self.hidden_dim,
            "num_bins": self.num_bins,
            "activations": self.activations,
        }

    def post_update(self, model: Model) -> Model:
        return l2normalize_flashsac_network(model)
