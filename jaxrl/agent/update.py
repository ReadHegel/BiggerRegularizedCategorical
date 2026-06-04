import functools

import jax
import jax.numpy as jnp

from typing import Tuple

from jaxrl.agent.networks.common import build_actor_input
from jaxrl.utils import Batch, Model, Params, PRNGKey, tree_norm

def update_actor(key: PRNGKey, actor: Model, critic: Model, temp: Model, batch: Batch, num_bins: int, v_max: float, multitask: bool):
    is_stateful = (actor.batch_stats is not None)
    if is_stateful:
        inputs_curr = build_actor_input(critic, batch.observations, batch.task_ids, multitask)
        inputs_next = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
        inputs_all = jnp.concatenate([inputs_curr, inputs_next], axis=0)
    else:
        inputs_all = build_actor_input(critic, batch.observations, batch.task_ids, multitask)

    def actor_loss_fn(actor_params: Params, batch_stats: Params = None):
        dist, actor_state_updates = actor.apply(actor_params, batch_stats, inputs_all, mutable=["batch_stats"], training=True)        
        if is_stateful:
            actions_all, log_probs_all = dist.sample_and_log_prob(seed=key)
            actions = jnp.split(actions_all, 2, axis=0)[0]
            log_probs = jnp.split(log_probs_all, 2, axis=0)[0]
        else:
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            
        q_logits, _ = critic(batch.observations, actions, batch.task_ids, mutable="batch_stats", training=False)        
        q_probs = jax.nn.softmax(q_logits, axis=-1).mean(axis=0)
        bin_values = jnp.linspace(start=-v_max, stop=v_max, num=num_bins)[None]
        q_values = (bin_values * q_probs).sum(-1)    
        actor_loss = (log_probs * temp().mean() - q_values).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            "actor_batch_stats": actor_state_updates.get("batch_stats"),
            'actor_pnorm': tree_norm(actor_params),
        }
    new_actor, info = actor.apply_gradient(actor_loss_fn)
    new_actor = new_actor.post_update()

    info['actor_gnorm'] = info.pop('grad_norm')

    new_actor = new_actor.replace(batch_stats=info.pop("actor_batch_stats"))

    return new_actor, info

def categorical_td_loss(
    next_log_probs: jnp.ndarray,  # (n, num_bins)
    next_q_logits: jnp.ndarray,  # (n, num_bins)
    batch: Batch,
    discount: float,
    num_bins: int,
    v_max: float,
    temp: Model,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    next_q_probs = jax.nn.softmax(next_q_logits, axis=-1).mean(axis=0)
    v_min = -v_max
    bin_values = jnp.linspace(start=v_min, stop=v_max, num=num_bins)[None]
    
    delta_z = ((v_max - v_min) / (num_bins - 1))
    target_bin_values = batch.rewards[:, None] + discount * batch.masks[:, None] * (bin_values - temp() * next_log_probs[:, None])
    target_bin_values = jnp.clip(target_bin_values, v_min, v_max)
    target_bin_values = (target_bin_values - v_min) / delta_z
    
    lower, upper = jnp.floor(target_bin_values), jnp.ceil(target_bin_values)
    lower_mask = jax.nn.one_hot(lower.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    upper_mask = jax.nn.one_hot(upper.reshape(-1), num_bins).reshape((-1, num_bins, num_bins))
    
    lower_values = (next_q_probs * (upper + (lower == upper).astype(jnp.float32) - target_bin_values))[..., None]        
    upper_values = (next_q_probs * (target_bin_values - lower))[..., None]
    
    target_probs = jax.lax.stop_gradient(jnp.sum(lower_values * lower_mask + upper_values * upper_mask, axis=1))
    q_value_target = (bin_values * target_probs).sum(-1)

    return target_probs, q_value_target

def update_critic(key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float, num_bins: int, v_max: float, multitask: bool):
    inputs = build_actor_input(critic, batch.next_observations, batch.task_ids, multitask)
    dist = actor(inputs, training=False)
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    
    is_stateful = (critic.batch_stats is not None)
    if is_stateful:
        obs_all = jnp.concatenate([batch.observations, batch.next_observations], axis=0)
        act_all = jnp.concatenate([batch.actions, next_actions], axis=0)
        task_ids_all = jnp.concatenate([batch.task_ids, batch.task_ids], axis=0) if batch.task_ids is not None else None
        
        target_q_logits_all, _ = target_critic(
            obs_all,
            act_all,
            task_ids_all,
            mutable="batch_stats",
            training=True,
        )
        next_q_logits = jnp.split(target_q_logits_all, 2, axis=1)[1]
    else:
        obs_all = None
        act_all = None
        task_ids_all = None
        next_q_logits, _ = target_critic(
            batch.next_observations,
            next_actions,
            batch.task_ids,
            mutable="batch_stats",
            training=True, # I am not sure if True is correct here, but it how it is done in the original code, also 
                        # the running batch stats are not updated at all for target critic, so trainging=False would be wrong
        )    

    target_probs, q_value_target = categorical_td_loss(next_log_probs, next_q_logits, batch, discount, num_bins, v_max, temp)

    def critic_loss_fn(critic_params: Params, batch_stats: Params = None):
        if is_stateful:
            q_logits_all, critic_state_updates = critic.apply(
                critic_params,
                batch_stats,
                obs_all,
                act_all,
                task_ids_all,
                mutable="batch_stats",
                training=True,
            )
            q_logits = jnp.split(q_logits_all, 2, axis=1)[0]
        else:
            q_logits, critic_state_updates = critic.apply(
                critic_params,
                batch_stats,
                batch.observations,
                batch.actions,
                batch.task_ids,
                mutable="batch_stats",
                training=True,
            )
        q_logprobs = jax.nn.log_softmax(q_logits, axis=-1)
        critic_loss = -(target_probs[None] * q_logprobs).sum(-1).mean(-1).sum(-1)
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q_value_target.mean(),
            "q_min": q_value_target.min(),
            "q_max": q_value_target.max(),
            "r": batch.rewards.mean(),
            "critic_pnorm": tree_norm(critic_params),
            "critic_batch_stats": critic_state_updates.get("batch_stats"),
        }
        
    new_critic, info = critic.apply_gradient(critic_loss_fn)
    new_critic = new_critic.post_update()

    info["critic_gnorm"] = info.pop("grad_norm")

    new_critic = new_critic.replace(batch_stats=info.pop("critic_batch_stats"))

    return new_critic, info

def update_target_critic(critic: Model, target_critic: Model, tau: float):
    new_target_params = jax.tree.map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)
    return target_critic.replace(params=new_target_params)

def update_temperature(temp: Model, entropy: float, target_entropy: float):
    def temperature_loss_fn(temp_params, batch_stats: Params = None):
        temperature = temp.apply(temp_params)
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {'temperature': temperature, 'temp_loss': temp_loss}
    new_temp, info = temp.apply_gradient(temperature_loss_fn)
    info.pop('grad_norm')
    return new_temp, info

'''
from jaxrl.utils import Batch

key = agent.rng
actor = agent.actor
target_critic = agent.target_critic
critic = agent.critic
temp = agent.temp
batch = Batch(
    observations=batches.observations[0],
    actions=batches.actions[0],
    rewards=batches.rewards[0],
    masks=batches.masks[0],
    next_observations=batches.next_observations[0],
    task_ids=batches.task_ids[0])
discount = agent.discount
num_bins = agent.num_bins
v_max = agent.v_max
multitask = agent.multitask
'''
