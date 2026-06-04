from __future__ import annotations

import functools
from typing import Callable, Optional, Tuple, Type

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from jaxrl.utils import Model


@functools.partial(jax.jit, static_argnames=('multitask',))
def build_actor_input(
    critic: Model,
    observations: jnp.ndarray,
    task_ids: jnp.ndarray,
    multitask: bool,
) -> jnp.ndarray:
    inputs = observations
    if multitask:
        task_embeddings = critic(None, None, task_ids, training=True, return_embeddings=True)
        inputs = jnp.concatenate((inputs, task_embeddings), axis=-1)
    return inputs


class TaskEmbedding(nn.Module):
    num_tasks: int
    embedding_size: int
    
    def setup(self):
        self.embeddings = nn.Embed(self.num_tasks, self.embedding_size)
        
    def __call__(self, x: jnp.ndarray):
        emb = self.embeddings(x)
        norm = jnp.linalg.norm(emb, axis=-1, keepdims=True)
        emb = emb/norm # TODO maybe add epislon ? 
        return emb
        
class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            'log_temp',
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


class BaseActor(nn.Module):
    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> distrax.Distribution:
        raise NotImplementedError(
            "Zaimplementuj __call__ jak NormalTanhPolicy: backbone -> distrax + Tanh."
        )

    def post_update(self, model: Model) -> Model:
        return model


class BaseEnsembleMultitaskCritic(nn.Module):
    num_tasks: int
    embedding_size: int
    ensemble_size: int = 2
    multitask: bool = False

    q_module = None

    def q_member_kwargs(self) -> dict:
        raise NotImplementedError

    def setup(self):
        if self.multitask:
            self.task_embedding = TaskEmbedding(self.num_tasks, self.embedding_size)
        VmapCritic = nn.vmap(
            self.q_module,
            variable_axes={'params': 0, "batch_stats": 0},
            split_rngs={'params': True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size,
        )
        self.q_value_ensemble = VmapCritic(**self.q_member_kwargs())

    def __call__(
        self,
        observations: Optional[jnp.ndarray],
        actions: Optional[jnp.ndarray],
        task_ids: jnp.ndarray,
        training: bool = False,
        return_embeddings: bool = False,
    ) -> jnp.ndarray:
        if self.multitask is False:
            inputs = jnp.concatenate((observations, actions), axis=-1)
        else:
            task_embedding = self.task_embedding(task_ids)
            if return_embeddings:
                return task_embedding
            inputs = jnp.concatenate((observations, actions, task_embedding), axis=-1)            
        q_values = self.q_value_ensemble(inputs, training)
        return q_values

    def post_update(self, model: Model) -> Model:
        return model

def build_actor_critic(
    arch: str,
    *,
    action_dim: int,
    num_tasks: int,
    embedding_size: int,
    ensemble_size: int,
    num_bins: int,
    multitask: bool,
    actor_overrides: dict | None = None,
    critic_overrides: dict | None = None,
) -> Tuple[nn.Module, nn.Module]:
    from jaxrl.agent.networks.BRO.networks import make_bro_actor, make_bro_critic

    actor_overrides = actor_overrides or {}
    critic_overrides = critic_overrides or {}

    if arch == "bro":
        actor = make_bro_actor(action_dim, **actor_overrides)
        critic = make_bro_critic(
            num_tasks=num_tasks,
            embedding_size=embedding_size,
            ensemble_size=ensemble_size,
            num_bins=num_bins,
            multitask=multitask,
            **critic_overrides,
        )
        return actor, critic
    elif arch == "simbaV2":
        from jaxrl.agent.networks.SimbaV2.simbaV2_network import make_simba_actor, make_simba_critic
        actor = make_simba_actor(action_dim, **actor_overrides)
        critic = make_simba_critic(
            num_tasks=num_tasks,
            embedding_size=embedding_size,
            ensemble_size=ensemble_size,
            num_bins=num_bins,
            multitask=multitask,
            **critic_overrides,
        )
        return actor, critic
    elif arch == "xqc":
        from jaxrl.agent.networks.XQC.networks import make_xqc_actor, make_xqc_critic
        actor = make_xqc_actor(action_dim, **actor_overrides)
        critic = make_xqc_critic(
            num_tasks=num_tasks,
            embedding_size=embedding_size,
            ensemble_size=ensemble_size,
            num_bins=num_bins,
            multitask=multitask,
            **critic_overrides,
        )
        return actor, critic
    elif arch == "flashsac":
        from jaxrl.agent.networks.FlashSAC.flashsac_network import make_flashsac_actor, make_flashsac_critic
        actor = make_flashsac_actor(action_dim, **actor_overrides)
        critic = make_flashsac_critic(
            num_tasks=num_tasks,
            embedding_size=embedding_size,
            ensemble_size=ensemble_size,
            num_bins=num_bins,
            multitask=multitask,
            **critic_overrides,
        )
        return actor, critic
    raise ValueError(f"Unsupported architecture: {arch!r}")
