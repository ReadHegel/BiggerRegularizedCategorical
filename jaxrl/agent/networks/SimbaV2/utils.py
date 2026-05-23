import re
from typing import Any, Dict, Tuple

import flax
import jax
import jax.numpy as jnp

from jaxrl.utils import Model

EPS = 1e-8

def tree_map_until_match(
    f, tree, target_re, *rest, keep_structure=True, keep_values=False
):
    """
    Similar to `jax.tree_util.tree_map_with_path`, but `is_leaf` is a regex condition.
    args:
        f: A function to map the discovered nodes (i.e., dict key matches `target_re`).
           Inputs to f will be (1) the discovered node and (2) the corresponding nodes in `*rest``.
        target_re: A regex string condition that triggers `f`.
        tree: A pytree to be searched by `target_re` and mapped by `f`.
        *rest: List of pytrees that are at least 'almost' identical structure to `tree`.
               'Almost', since the substructure of matching nodes don't have to be identical.
               i.e., The tree structure of `tree` and `*rest` should be identical only up to the matching nodes.
        keep_structure: If false, the returned tree will only contain subtrees that lead to the matching nodes.
        keep_values: If false, unmatched leaves will become `None`. Assumes `keep_structure=True`.
    """

    if not isinstance(tree, dict):
        return tree if keep_values else None

    ret_tree = {}
    for k, v in tree.items():
        v_rest = [r[k] for r in rest]
        if re.fullmatch(target_re, k):
            ret_tree[k] = f(v, *v_rest)
        else:
            subtree = tree_map_until_match(
                f,
                v,
                target_re,
                *v_rest,
                keep_structure=keep_structure,
                keep_values=keep_values,
            )
            if keep_structure or subtree:
                ret_tree[k] = subtree

    return ret_tree

def l2normalize(
    x: jnp.ndarray,
    axis: int,
) -> jnp.ndarray:
    l2norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    x = x / jnp.maximum(l2norm, EPS)

    return x


def l2normalize_layer(tree):
    """
    apply l2-normalization to the all leaf nodes
    """
    if len(tree["kernel"].shape) == 2:
        axis = 0
    elif len(tree["kernel"].shape) == 3:
        axis = 1
    else:
        raise ValueError
    return jax.tree.map(f=lambda x: l2normalize(x, axis=axis), tree=tree)


def l2normalize_network(
    network: Model,
    regex: str = "hyper_dense",
) -> Model:
    params = network.params
    new_params = tree_map_until_match(
        f=lambda x: l2normalize_layer(x), tree=params, target_re=regex, keep_values=True
    )
    network = network.replace(params=new_params)
    return network