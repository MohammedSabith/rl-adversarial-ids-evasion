"""
Base Gymnasium environment for RL evasion of ML classifiers.

Subclass this to plug in your own classifier + dataset.
Implement _apply_action(), _recalculate_derived(), and _is_valid().
"""

from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BaseEvasionEnv(gym.Env, ABC):
    """Abstract base for RL classifier evasion environments.

    Provides the generic RL loop (reset/step/normalize).
    Subclasses implement domain-specific action application,
    derived feature recalculation, and validity constraints.
    """

    metadata = {'render_modes': []}

    def __init__(self, clf, flow_pool, n_features, n_actions, benign_class, max_steps):
        """
        Args:
            clf: Classifier with a sklearn-compatible predict_proba() interface.
            flow_pool: np.ndarray of shape (n_flows, n_features) — malicious flows.
            n_features: Number of raw features per flow.
            n_actions: Size of the discrete action space.
            benign_class: Index of the benign class in predict_proba() output.
            max_steps: Maximum modification steps per episode.
        """
        super().__init__()

        self.clf = clf
        self.flow_pool = flow_pool
        self.benign_class = benign_class
        self.max_steps = max_steps

        # Observation normalization (z-score)
        self.obs_mean = self.flow_pool.mean(axis=0).astype(np.float64)
        self.obs_std = self.flow_pool.std(axis=0).astype(np.float64)
        self.obs_std[self.obs_std < 1e-8] = 1.0

        # Spaces: n_features normalized + step fraction
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(n_features + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)

        # Episode state
        self.current_flow = None
        self.original_flow = None
        self.steps_taken = 0
        self.prev_prob_malicious = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(0, len(self.flow_pool))
        self.current_flow = self.flow_pool[idx].copy().astype(np.float64)
        self.original_flow = self.current_flow.copy()
        self.steps_taken = 0

        proba = self.clf.predict_proba([self.current_flow])[0]
        self.prev_prob_malicious = 1.0 - proba[self.benign_class]

        return self._normalize(self.current_flow), {}

    def step(self, action):
        assert self.current_flow is not None, "Call reset() before step()"
        self.steps_taken += 1

        modified = self._apply_action(self.current_flow, action)
        modified = self._recalculate_derived(modified)

        if not self._is_valid(modified):
            self.current_flow = modified
            return self._normalize(modified), -0.5, True, False, {
                'evaded': False, 'reason': 'invalid_flow', 'step': self.steps_taken
            }

        self.current_flow = modified

        proba = self.clf.predict_proba([modified])[0]
        prob_malicious = 1.0 - proba[self.benign_class]
        evaded = proba[self.benign_class] > 0.5

        is_final = self.steps_taken >= self.max_steps
        reward = self.prev_prob_malicious - prob_malicious
        if evaded:
            reward += 1.0
        elif is_final:
            reward -= 1.0

        self.prev_prob_malicious = prob_malicious
        terminated = is_final or evaded

        return self._normalize(self.current_flow), reward, terminated, False, {
            'evaded': evaded, 'prob_malicious': prob_malicious, 'step': self.steps_taken,
        }

    def _normalize(self, flow):
        norm = (flow - self.obs_mean) / self.obs_std
        step_norm = self.steps_taken / self.max_steps
        return np.append(norm, step_norm).astype(np.float32)

    @abstractmethod
    def _apply_action(self, flow, action):
        """Apply a stochastic action to the flow. Return modified flow."""

    @abstractmethod
    def _recalculate_derived(self, flow):
        """Recalculate derived features from base modifications. Return flow.

        self.original_flow contains the unmodified flow from episode start."""

    @abstractmethod
    def _is_valid(self, flow):
        """Check if the modified flow satisfies constraints. Return bool."""
