import numpy as np

from jaxrl.utils import Batch

# RunningMeanStd adapted from SimbaV2 (Apache 2.0):
# external_repos/SimbaV2/scale_rl/agents/wrappers/utils.py


def _update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count
    ratio = batch_count / tot_count
    new_mean = mean + delta * ratio
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * count * ratio
    new_var = m2 / tot_count
    return new_mean, new_var, tot_count


class RunningMeanStd:
    """Tracks per-feature mean, variance and count (Welford-style parallel merge)."""

    def __init__(self, shape=(), dtype=np.float32, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=self.mean.dtype)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.mean, self.var, self.count = _update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


class ObservationNormalizer:
    """
    Online observation whitening matching SimbaV2 ObservationNormalizer:
    - update running stats only during interaction (training=True)
    - normalize replay batches at update time without updating stats
    """

    def __init__(self, obs_dim: int, epsilon: float = 1e-8):
        self.obs_rms = RunningMeanStd(shape=(obs_dim,), dtype=np.float32)
        self.epsilon = epsilon

    def update(self, observations: np.ndarray) -> None:
        self.obs_rms.update(observations)

    def normalize(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        return (observations - self.obs_rms.mean) / np.sqrt(
            self.obs_rms.var + self.epsilon
        )

    def normalize_batch(self, batch: Batch) -> Batch:
        return Batch(
            observations=self.normalize(batch.observations),
            actions=batch.actions,
            rewards=batch.rewards,
            masks=batch.masks,
            next_observations=self.normalize(batch.next_observations),
            task_ids=batch.task_ids,
        )


def observation_normalizer_for_arch(arch: str, obs_dim: int) -> ObservationNormalizer | None:
    if arch == "simbaV2":
        return ObservationNormalizer(obs_dim=obs_dim)
    return None


class RewardNormalizer(object):
    def __init__(self, num_seeds: int, target_entropy: float, discount: float = 0.99, v_max: float = 10.0, max_steps: int | None = None):
        self.returns_min_norm = np.zeros(num_seeds, dtype=np.float32) + np.inf
        self.returns_max_norm = np.zeros(num_seeds, dtype=np.float32) - np.inf           
        self.effective_horizon = 1 / (1 - discount)
        self.discount = discount
        self.v_max = v_max
        self.target_entropy = target_entropy        
        self.max_steps = max_steps
        self.step = 0
        self.rewards = np.zeros((num_seeds, max_steps), dtype=np.float32) if max_steps is not None else [[] for _ in range(num_seeds)]
        
    def _calculate_returns_variable_length_trajectory(self, rewards_traj: list, truncate: bool):
        values = np.zeros_like(rewards_traj)
        bootstrap = rewards_traj.mean() * self.effective_horizon if truncate else 0.0
        for i in reversed(range(rewards_traj.shape[0])):
            values[i] = rewards_traj[i] + self.discount * bootstrap
            bootstrap = values[i]
        return values.min(axis=-1), values.max(axis=-1)
    
    def _calculate_returns_fixed_length_trajectory(self):
        values = np.zeros_like(self.rewards, dtype=np.float32)
        bootstrap = self.rewards.mean(-1) * self.effective_horizon
        for i in reversed(range(values.shape[-1])):
            values[:, i] = self.rewards[:, i] + self.discount * bootstrap
            bootstrap = values[:, i]
        return values.min(axis=-1), values.max(axis=-1)
        
    def _update_variable_length_trajectory(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray):
        for i, reward in enumerate(rewards):
            self.rewards[i].append(reward)
        done = np.logical_or(terminal, truncate)
        if done.any():
            indx = done.nonzero()[0]
            for j in indx:
                rewards_traj = np.asarray(self.rewards[j])
                value_min, value_max = self._calculate_returns_variable_length_trajectory(rewards_traj, truncate[j])
                self.returns_min_norm[j] = min(self.returns_min_norm[j], value_min) 
                self.returns_max_norm[j] = max(self.returns_max_norm[j], value_max) 
                self.rewards[j] = []
                
    def _update_fixed_length_trajectory(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray):
        self.rewards[:, self.step] = rewards
        dones = np.logical_or(terminal, truncate)
        if self.step == self.max_steps - 1:
            assert dones.all()
            v_min, v_max = self._calculate_returns_fixed_length_trajectory()
            self.returns_min_norm = np.where(v_min < self.returns_min_norm, v_min, self.returns_min_norm)
            self.returns_max_norm = np.where(v_max > self.returns_max_norm, v_max, self.returns_max_norm)            
            self.step = 0
        else:
            self.step += 1
        
    def update(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray):
        if self.max_steps is not None:
            self._update_fixed_length_trajectory(rewards, terminal, truncate)
        else:
            self._update_variable_length_trajectory(rewards, terminal, truncate)
            
    def normalize(self, batches: Batch, temperature: np.ndarray):
        denominator = np.where(self.returns_max_norm > np.abs(self.returns_min_norm), self.returns_max_norm, np.abs(self.returns_min_norm))
        denominator = (denominator - temperature * self.effective_horizon * self.target_entropy / 2) / self.v_max
        denominator = denominator[batches.task_ids]
        rewards = batches.rewards / denominator
        return Batch(observations=batches.observations, actions=batches.actions, rewards=rewards, masks=batches.masks, next_observations=batches.next_observations, task_ids=batches.task_ids)
   