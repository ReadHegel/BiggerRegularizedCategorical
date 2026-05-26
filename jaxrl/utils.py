import os
import collections
from typing import Any, Optional, Sequence

import flax
import jax
import jax.numpy as jnp
import optax

Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations', 'task_ids'])

def tree_norm(tree):
    return jnp.sqrt(sum((x**2).sum() for x in jax.tree_util.tree_leaves(tree)))

@flax.struct.dataclass
class SaveState:
    params: Params
    opt_state: Optional[optax.OptState] = None


@flax.struct.dataclass
class Model:
    step: int
    apply_fn: flax.linen.Module = flax.struct.field(pytree_node=False)
    params: Params
    batch_stats: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None


    @classmethod
    def create(cls,
               model_def: flax.linen.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None):
        variables = model_def.init(*inputs)

        params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def,
                   params=params,
                   tx=tx,
                   opt_state=opt_state,
                   batch_stats=variables.get('batch_stats', None),
                )

    def __call__(self, *args, **kwargs):
        variables = {"params": self.params}
        if self.batch_stats is not None:
            variables["batch_stats"] = self.batch_stats
        return self.apply_fn.apply(variables, *args, **kwargs)

    def apply(self, params, batch_stats=None, *args, **kwargs):
        variables = {
            "params": params,
        }
        if batch_stats is not None:
            variables["batch_stats"] = batch_stats
        return self.apply_fn.apply(variables, *args, **kwargs)

    def apply_gradient(self, loss_fn):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params, self.batch_stats)
        grad_norm = tree_norm(grads)
        info['grad_norm'] = grad_norm

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info

    def post_update(self):
        return self.apply_fn.post_update(self)
    
    def get_gradient(self, loss_fn):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)
        return grads

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(
                flax.serialization.to_bytes(
                    SaveState(
                        params=self.params, 
                        opt_state=self.opt_state, 
                        batch_stats=self.batch_stats,
                    )
                )
            )

    def load(self, load_path: str):
        with open(load_path, 'rb') as f:
            contents = f.read()
            saved_state = flax.serialization.from_bytes(
                SaveState(
                    params=self.params,
                    opt_state=self.opt_state,
                    batch_stats=self.batch_stats,
                ),
                contents,
            )
        return self.replace(
            params=saved_state.params,
            opt_state=saved_state.opt_state,
            batch_stats=saved_state.batch_stats,
        )