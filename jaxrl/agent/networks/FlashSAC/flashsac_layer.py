import distrax
import flax.linen as nn
import jax.numpy as jnp


class UnitLinear(nn.Module):
    output_dim: int 

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(
            features=self.output_dim,
            use_bias=False,
            kernel_init=nn.initializers.orthogonal(scale=1.0, column_axis=0),
            name="unit_linear",
        )(x)

# TODO It's stateless. Not as in the original paper.
class UnitBatchNorm(nn.Module):
    momentum: float = 0.01
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        input_dim = x.shape[-1]
        # Weight and bias parameters are normalized in l2normalize_flashsac_network function
        weight = self.param("weight", nn.initializers.ones, (input_dim,))
        bias = self.param("bias", nn.initializers.zeros, (input_dim,))

        # **Stateless** mean and variance across the batch dimension
        mean = jnp.mean(x, axis=0, keepdims=True)
        var = jnp.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + self.eps)
        return x_norm * weight + bias


class UnitRMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_dim = x.shape[-1]
        # Weight parameter is normalized in l2normalize_flashsac_network function
        weight = self.param("weight", nn.initializers.ones, (input_dim,))
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return (x / rms) * weight


class FlashSACEmbedder(nn.Module):
    input_dim: int
    hidden_dim: int
    norm_type: str = "rms_norm" # TODO original paper uses UnitBatchNorm

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        if self.norm_type == "batch_norm":
            x = UnitBatchNorm(name="unit_batch_norm")(x, training=training)
            x = UnitLinear(output_dim=self.hidden_dim, name="w")(x)
        elif self.norm_type == "rms_norm":
            x = UnitRMSNorm(name="unit_rms_norm")(x)
            x = UnitLinear(output_dim=self.hidden_dim, name="w")(x)
        return x


class FlashSACBlock(nn.Module):
    hidden_dim: int
    expansion: int = 4
    norm_type: str = "rms_norm" # TODO original paper uses UnitBatchNorm

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        residual = x
        if self.norm_type == "batch_norm":
            x = UnitLinear(output_dim=self.hidden_dim * self.expansion, name="w1")(x)
            x = UnitBatchNorm(name="unit_batch_norm1")(x, training=training)
            x = nn.relu(x)
            x = UnitLinear(output_dim=self.hidden_dim, name="w2")(x)
            x = UnitBatchNorm(name="unit_batch_norm2")(x, training=training)
            x = nn.relu(x)
        elif self.norm_type == "rms_norm":
            x = UnitLinear(output_dim=self.hidden_dim * self.expansion, name="w1")(x)
            x = UnitRMSNorm(name="unit_rms_norm1")(x)
            x = nn.relu(x)
            x = UnitLinear(output_dim=self.hidden_dim, name="w2")(x)
            x = UnitRMSNorm(name="unit_rms_norm2")(x)
            x = nn.relu(x)
        return x + residual


class NormalTanhPolicy(nn.Module):
    hidden_dim: int
    action_dim: int
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        temperature: float = 1.0,
    ) -> distrax.Distribution:
        mean_w = UnitLinear(output_dim=self.action_dim, name="mean_w")
        mean_bias = self.param("mean_bias", nn.initializers.zeros, (self.action_dim,))

        std_w = UnitLinear(output_dim=self.action_dim, name="std_w")
        std_bias = self.param("std_bias", nn.initializers.zeros, (self.action_dim,))

        mean = mean_w(x) + mean_bias
        raw_log_std = std_w(x) + std_bias

        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + jnp.tanh(raw_log_std))
        std = jnp.exp(log_std)
        stds = std * temperature
        base_dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=stds)
        return distrax.Transformed(base_dist, distrax.Block(distrax.Tanh(), 1))


class CategoricalValue(nn.Module):
    hidden_dim: int
    num_bins: int

    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        w = UnitLinear(output_dim=self.num_bins, name="w")
        bias = self.param("bias", nn.initializers.zeros, (self.num_bins,))
        return w(x) + bias
