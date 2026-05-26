import flax.traverse_util
import jax.numpy as jnp

from jaxrl.utils import Model

def norm_dense_layer(params, path, norm_bias=True):

    kernel = params[path + '/kernel']
    bias = params.get(path + '/bias', None)
    
    # if bias is present, normalize kernel and bias together
    if norm_bias and bias is not None:
        w = jnp.concatenate([kernel, jnp.expand_dims(bias, -2)], axis=-2)
    else:
        w = kernel
    
    norm = jnp.linalg.norm(w, axis=-2, keepdims=True)    
    
    params[path + '/kernel'] = kernel / norm
    if norm_bias and bias is not None:
        params[path + '/bias'] = bias / norm.squeeze(-2)

    return params


def norm_network(
    model: Model,
    normalize_last_layer: bool =True,
):
    params_flat = flax.traverse_util.flatten_dict(model.params, sep="/")

    for path in sorted({'/'.join(k.split('/')[:-1]) for k in params_flat}):
        
        # Normalize all hidden Dense layers, i.e. under 'MLP_0'
        if 'MLP_0' in path and 'Dense' in path:
            params_flat = norm_dense_layer(params_flat, path, norm_bias=True)
        
        # Normalize last Dense layer, i.e. under 'predictor'
        elif 'predictor' in path and normalize_last_layer:
            params_flat = norm_dense_layer(params_flat, path, norm_bias=False)
    
    return model.replace(params=flax.traverse_util.unflatten_dict(params_flat, sep="/"))