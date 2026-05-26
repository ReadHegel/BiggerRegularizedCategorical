import flax.linen as nn
import jax.numpy as jnp
import distrax

from jaxrl.agent.networks.SimbaV2.simbaV2_layer import (
    HyperCategoricalValue,
    HyperEmbedder,
    HyperLERPBlock,
    HyperNormalTanhPolicy,
)

from jaxrl.agent.networks.common import BaseActor, BaseEnsembleMultitaskCritic
from jaxrl.agent.networks.SimbaV2.utils import l2normalize_network
from jaxrl.utils import Model

SIMBA_ACTOR_CONFIG = {
    "num_blocks": 2,
    "hidden_dim": 256,
    "scaler_init": 1.0,
    "scaler_scale": 1.0,
    "alpha_init": 0.3,
    "alpha_scale": 1.0,
    "c_shift": 3.0,
}

SIMBA_CRITIC_CONFIG = {
    "num_blocks": 4,
    "hidden_dim": 512,
    "scaler_init": 1.0,
    "scaler_scale": 1.0,
    "alpha_init": 0.3,
    "alpha_scale": 1.0,
    "c_shift": 3.0,
}


def make_simba_actor(action_dim: int, **overrides) -> "SimbaV2Actor":
    cfg = {**SIMBA_ACTOR_CONFIG, **overrides, "action_dim": action_dim}
    return SimbaV2Actor(**cfg)
    
def make_simba_critic(
    *,
    num_tasks: int,
    embedding_size: int,
    ensemble_size: int,
    num_bins: int,
    multitask: bool,
    **overrides,
) -> "SimbaV2Critic":
    cfg = {
        **SIMBA_CRITIC_CONFIG,
        **overrides,
        "num_tasks": num_tasks,
        "embedding_size": embedding_size,
        "ensemble_size": ensemble_size,
        "num_bins": num_bins,
        "multitask": multitask,
    }
    return SimbaV2Critic(**cfg)

class SimbaV2Actor(BaseActor):
    num_blocks: int
    hidden_dim: int
    action_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float

    def setup(self):
        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.predictor = HyperNormalTanhPolicy(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
        training: bool = False, # ignore 
    ) -> distrax.Distribution:
        x = observations
        y = self.embedder(x)
        z = self.encoder(y)
        dist = self.predictor(z, temperature)

        return dist

    def post_update(self, model: Model) -> Model:
        return l2normalize_network(model)


class SimbaV2QValue(nn.Module):
    num_blocks: int
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    num_bins: int

    def setup(self):
        self.embedder = HyperEmbedder(
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.predictor = HyperCategoricalValue(
            hidden_dim=self.hidden_dim,
            num_bins=self.num_bins,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        training: bool, # ignore 
    ) -> jnp.ndarray:
        y = self.embedder(inputs)
        z = self.encoder(y)
        logits = self.predictor(z)
        return logits

class SimbaV2Critic(BaseEnsembleMultitaskCritic):
    q_module = SimbaV2QValue

    num_blocks: int = 4
    hidden_dim: int = 512
    scaler_init: float = 1.0
    scaler_scale: float = 1.0
    alpha_init: float = 0.3
    alpha_scale: float = 1.0
    c_shift: float = 3.0
    num_bins: int = 101

    def q_member_kwargs(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "hidden_dim": self.hidden_dim,
            "scaler_init": self.scaler_init,
            "scaler_scale": self.scaler_scale,
            "alpha_init": self.alpha_init,
            "alpha_scale": self.alpha_scale,
            "c_shift": self.c_shift,
            "num_bins": self.num_bins,
        }

    def post_update(self, model: Model) -> Model:
        return l2normalize_network(model)