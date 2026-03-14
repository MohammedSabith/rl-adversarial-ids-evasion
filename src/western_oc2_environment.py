"""
Gymnasium environment for RL evasion of Western-OC2-Lab's CICIDS2017 Random Forest.

Externally validated classifier:
  - 573 GitHub stars, 3 IEEE papers (GLOBECOM 2019, IoT Journal 2022, GLOBECOM 2022)
  - Random Forest, 99.41% accuracy (reproduced from their exact notebook)
  - 77 CICFlowMeter features, 7 attack classes

The attacker modifies 3 base forward features. All other features are
recalculated, proportionally scaled, or kept fixed (backward/flags/timing).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import joblib
import os

# -----------------------------------------------------------------------
# Feature indices (must match feature_names.npy from Western-OC2-Lab)
# -----------------------------------------------------------------------
I = {
    'flow_duration': 0,
    'total_fwd_packets': 1, 'total_bwd_packets': 2,
    'total_len_fwd_packets': 3, 'total_len_bwd_packets': 4,
    'fwd_pkt_len_max': 5, 'fwd_pkt_len_min': 6,
    'fwd_pkt_len_mean': 7, 'fwd_pkt_len_std': 8,
    'bwd_pkt_len_max': 9, 'bwd_pkt_len_min': 10,
    'bwd_pkt_len_mean': 11, 'bwd_pkt_len_std': 12,
    'flow_bytes_s': 13, 'flow_packets_s': 14,
    'flow_iat_mean': 15, 'flow_iat_std': 16,
    'flow_iat_max': 17, 'flow_iat_min': 18,
    'fwd_iat_total': 19, 'fwd_iat_mean': 20,
    'fwd_iat_std': 21, 'fwd_iat_max': 22, 'fwd_iat_min': 23,
    'bwd_iat_total': 24, 'bwd_iat_mean': 25,
    'bwd_iat_std': 26, 'bwd_iat_max': 27, 'bwd_iat_min': 28,
    'fwd_psh_flags': 29, 'bwd_psh_flags': 30,
    'fwd_urg_flags': 31, 'bwd_urg_flags': 32,
    'fwd_header_length': 33, 'bwd_header_length': 34,
    'fwd_packets_s': 35, 'bwd_packets_s': 36,
    'min_packet_length': 37, 'max_packet_length': 38,
    'packet_length_mean': 39, 'packet_length_std': 40,
    'packet_length_variance': 41,
    'fin_flag': 42, 'syn_flag': 43, 'rst_flag': 44,
    'psh_flag': 45, 'ack_flag': 46, 'urg_flag': 47,
    'cwe_flag': 48, 'ece_flag': 49,
    'down_up_ratio': 50,
    'avg_packet_size': 51, 'avg_fwd_seg_size': 52, 'avg_bwd_seg_size': 53,
    'fwd_header_length_1': 54,
    'fwd_avg_bytes_bulk': 55, 'fwd_avg_packets_bulk': 56, 'fwd_avg_bulk_rate': 57,
    'bwd_avg_bytes_bulk': 58, 'bwd_avg_packets_bulk': 59, 'bwd_avg_bulk_rate': 60,
    'subflow_fwd_packets': 61, 'subflow_fwd_bytes': 62,
    'subflow_bwd_packets': 63, 'subflow_bwd_bytes': 64,
    'init_win_bytes_fwd': 65, 'init_win_bytes_bwd': 66,
    'act_data_pkt_fwd': 67, 'min_seg_size_fwd': 68,
    'active_mean': 69, 'active_std': 70,
    'active_max': 71, 'active_min': 72,
    'idle_mean': 73, 'idle_std': 74,
    'idle_max': 75, 'idle_min': 76,
}
N_FEATURES = 77
BENIGN_CLASS = 0  # LabelEncoder: BENIGN=0

# -----------------------------------------------------------------------
# Action definitions: 12 discrete stochastic actions (no protocol switch)
# -----------------------------------------------------------------------
ACTION_DEFS = [
    ('total_fwd_packets',      +1, 0.25, 0.05),   # 0: fwd pkts +25%
    ('total_fwd_packets',      +1, 0.10, 0.025),   # 1: fwd pkts +10%
    ('total_fwd_packets',      -1, 0.10, 0.025),   # 2: fwd pkts -10%
    ('total_fwd_packets',      -1, 0.25, 0.05),    # 3: fwd pkts -25%
    ('total_len_fwd_packets',  +1, 0.25, 0.05),    # 4: fwd bytes +25%
    ('total_len_fwd_packets',  +1, 0.10, 0.025),   # 5: fwd bytes +10%
    ('total_len_fwd_packets',  -1, 0.10, 0.025),   # 6: fwd bytes -10%
    ('total_len_fwd_packets',  -1, 0.25, 0.05),    # 7: fwd bytes -25%
    ('fwd_iat_total',          +1, 0.25, 0.05),    # 8: fwd timing +25%
    ('fwd_iat_total',          +1, 0.10, 0.025),   # 9: fwd timing +10%
    ('fwd_iat_total',          -1, 0.10, 0.025),   # 10: fwd timing -10%
    ('fwd_iat_total',          -1, 0.25, 0.05),    # 11: fwd timing -25%
]

# Constraints
MIN_RATIO = 0.5
MAX_FWD_PKT_LEN = 1500  # MTU
MAX_STEPS = 5


class WesternOC2EvasionEnv(gym.Env):
    """RL environment: evade Western-OC2-Lab's peer-reviewed RF classifier."""

    metadata = {'render_modes': []}

    def __init__(self, max_steps=None):
        super().__init__()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models', 'western_oc2')

        self.clf = joblib.load(os.path.join(models_dir, 'rf_classifier.joblib'))
        all_flows = np.load(os.path.join(models_dir, 'malicious_flows.npy'))

        # Filter out tiny flows (<=40 bytes fwd payload = header-only or degenerate)
        # These inflate evasion rates because trivial byte changes (2→3) flip classification.
        fwd_bytes = all_flows[:, I['total_len_fwd_packets']]
        self.flow_pool = all_flows[fwd_bytes > 40]

        self.max_steps = max_steps if max_steps is not None else MAX_STEPS

        # Observation normalization
        self.obs_mean = self.flow_pool.mean(axis=0).astype(np.float64)
        self.obs_std = self.flow_pool.std(axis=0).astype(np.float64)
        self.obs_std[self.obs_std < 1e-8] = 1.0

        # Spaces: 77 normalized features + step number
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(N_FEATURES + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTION_DEFS))

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
        self.prev_prob_malicious = 1.0 - proba[BENIGN_CLASS]

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
        prob_malicious = 1.0 - proba[BENIGN_CLASS]
        evaded = proba[BENIGN_CLASS] > 0.5  # RF predicts BENIGN

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

    def _apply_action(self, flow, action):
        f = flow.copy()
        feat_name, direction, magnitude, noise_hw = ACTION_DEFS[action]

        feat_idx = I[feat_name]
        actual_mag = magnitude + self.np_random.uniform(-noise_hw, noise_hw)
        factor = 1.0 + direction * actual_mag
        new_val = f[feat_idx] * factor

        if feat_name == 'total_fwd_packets':
            if direction < 0:
                new_val = max(1, int(np.floor(new_val)))
            else:
                new_val = max(1, int(np.ceil(new_val)))
        elif feat_name == 'total_len_fwd_packets':
            # Bytes must be integer — round to prevent fractional byte artifacts
            new_val = max(1, round(new_val))
        else:
            new_val = max(0, new_val)

        f[feat_idx] = new_val
        return f

    def _recalculate_derived(self, flow):
        """Recalculate all features dependent on the 3 base modifiable features."""
        f = flow.copy()
        orig = self.original_flow

        fwd_pkts = f[I['total_fwd_packets']]
        fwd_bytes = f[I['total_len_fwd_packets']]
        fwd_iat_tot = f[I['fwd_iat_total']]
        bwd_pkts = f[I['total_bwd_packets']]      # fixed
        bwd_bytes = f[I['total_len_bwd_packets']]  # fixed
        bwd_iat_tot = f[I['bwd_iat_total']]        # fixed

        # --- Forward means (exact) ---
        old_fwd_mean = orig[I['fwd_pkt_len_mean']]
        new_fwd_mean = fwd_bytes / fwd_pkts if fwd_pkts > 0 else 0
        f[I['fwd_pkt_len_mean']] = new_fwd_mean
        f[I['avg_fwd_seg_size']] = new_fwd_mean  # same thing

        fwd_intervals = max(fwd_pkts - 1, 1)
        old_fwd_iat_mean = orig[I['fwd_iat_mean']]
        new_fwd_iat_mean = fwd_iat_tot / fwd_intervals
        f[I['fwd_iat_mean']] = new_fwd_iat_mean

        # --- Forward distribution stats (proportional scaling) ---
        if old_fwd_mean > 0 and new_fwd_mean > 0:
            bytes_scale = new_fwd_mean / old_fwd_mean
        else:
            bytes_scale = 1.0
        f[I['fwd_pkt_len_std']] = orig[I['fwd_pkt_len_std']] * bytes_scale
        f[I['fwd_pkt_len_max']] = orig[I['fwd_pkt_len_max']] * bytes_scale
        f[I['fwd_pkt_len_min']] = orig[I['fwd_pkt_len_min']] * bytes_scale
        f[I['min_seg_size_fwd']] = orig[I['min_seg_size_fwd']] * bytes_scale

        if old_fwd_iat_mean > 0 and new_fwd_iat_mean > 0:
            iat_scale = new_fwd_iat_mean / old_fwd_iat_mean
        else:
            iat_scale = 1.0
        f[I['fwd_iat_std']] = orig[I['fwd_iat_std']] * iat_scale
        f[I['fwd_iat_max']] = orig[I['fwd_iat_max']] * iat_scale
        f[I['fwd_iat_min']] = orig[I['fwd_iat_min']] * iat_scale

        # --- Forward other ---
        f[I['fwd_header_length']] = fwd_pkts * 32  # approx header bytes
        f[I['fwd_header_length_1']] = f[I['fwd_header_length']]  # exact duplicate
        orig_act_ratio = (orig[I['act_data_pkt_fwd']] / orig[I['total_fwd_packets']]
                          if orig[I['total_fwd_packets']] > 0 else 1)
        f[I['act_data_pkt_fwd']] = max(0, round(fwd_pkts * orig_act_ratio))

        # --- Subflow duplicates ---
        f[I['subflow_fwd_packets']] = fwd_pkts
        f[I['subflow_fwd_bytes']] = fwd_bytes

        # --- Flow Duration (additive, floored by bwd timing) ---
        delta_fwd_iat = fwd_iat_tot - orig[I['fwd_iat_total']]
        new_duration = max(bwd_iat_tot, orig[I['flow_duration']] + delta_fwd_iat)
        new_duration = max(0, new_duration)
        f[I['flow_duration']] = new_duration

        # --- Flow rates (depend on duration) ---
        total_bytes = fwd_bytes + bwd_bytes
        total_pkts = fwd_pkts + bwd_pkts
        if new_duration > 0:
            f[I['flow_bytes_s']] = total_bytes / new_duration * 1e6
            f[I['flow_packets_s']] = total_pkts / new_duration * 1e6
            f[I['fwd_packets_s']] = fwd_pkts / new_duration * 1e6
            f[I['bwd_packets_s']] = bwd_pkts / new_duration * 1e6
        else:
            f[I['flow_bytes_s']] = 0
            f[I['flow_packets_s']] = 0
            f[I['fwd_packets_s']] = 0
            f[I['bwd_packets_s']] = 0

        # --- Combined packet stats ---
        f[I['packet_length_mean']] = total_bytes / total_pkts if total_pkts > 0 else 0
        f[I['avg_packet_size']] = f[I['packet_length_mean']]
        f[I['min_packet_length']] = min(f[I['fwd_pkt_len_min']], f[I['bwd_pkt_len_min']])
        f[I['max_packet_length']] = max(f[I['fwd_pkt_len_max']], f[I['bwd_pkt_len_max']])

        # Combined Pkt Len Std/Var (exact formula)
        n1, n2 = fwd_pkts, bwd_pkts
        m1, m2 = new_fwd_mean, f[I['bwd_pkt_len_mean']]
        v1 = f[I['fwd_pkt_len_std']] ** 2
        v2 = f[I['bwd_pkt_len_std']] ** 2  # fixed
        m = f[I['packet_length_mean']]
        if n1 + n2 > 0:
            combined_var = (n1*v1 + n2*v2 + n1*(m1-m)**2 + n2*(m2-m)**2) / (n1+n2)
            f[I['packet_length_variance']] = max(0, combined_var)
            f[I['packet_length_std']] = np.sqrt(max(0, combined_var))
        else:
            f[I['packet_length_variance']] = 0
            f[I['packet_length_std']] = 0

        # --- Down/Up ratio ---
        f[I['down_up_ratio']] = bwd_pkts / fwd_pkts if fwd_pkts > 0 else 0

        # --- Flow IAT, Active/Idle, Flags, Bulk, Init Win: keep fixed ---

        return f

    def _is_valid(self, flow):
        """Check physical and functionality constraints."""
        fwd_pkts = flow[I['total_fwd_packets']]
        fwd_bytes = flow[I['total_len_fwd_packets']]
        fwd_iat = flow[I['fwd_iat_total']]

        if fwd_pkts < 1 or fwd_bytes < 0 or fwd_iat < 0:
            return False

        # MTU
        if fwd_pkts > 0:
            fwd_mean_pkt = fwd_bytes / fwd_pkts
            if fwd_mean_pkt > MAX_FWD_PKT_LEN:
                return False

        # Functionality constraints (min 50% of original)
        if self.original_flow is not None:
            orig = self.original_flow
            if orig[I['total_fwd_packets']] > 0 and fwd_pkts < orig[I['total_fwd_packets']] * MIN_RATIO:
                return False
            if orig[I['total_len_fwd_packets']] > 0 and fwd_bytes < orig[I['total_len_fwd_packets']] * MIN_RATIO:
                return False
            if orig[I['fwd_iat_total']] > 0 and fwd_iat < orig[I['fwd_iat_total']] * MIN_RATIO:
                return False

        return True
