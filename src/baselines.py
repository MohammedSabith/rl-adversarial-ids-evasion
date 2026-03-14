"""
Baseline evaluations for the Western-OC2-Lab evasion environment.

Provides fair comparisons for the RL agent:
  1. Random baseline: uniform action selection (same 12-action space)
  2. Exhaustive search: try all 12^3 = 1,728 deterministic 3-step sequences (upper bound)
"""

import numpy as np
from itertools import product
from src.western_oc2_environment import WesternOC2EvasionEnv, ACTION_DEFS, I, BENIGN_CLASS


def random_baseline(n_flows=1000, attempts_per_flow=1, seed=0):
    """Evaluate random uniform action selection.

    Args:
        n_flows: number of malicious flows to test
        attempts_per_flow: how many random 5-step sequences to try per flow
            1 = fair comparison with RL (single attempt)

    Returns:
        dict with evasion rate and per-flow results
    """
    env = WesternOC2EvasionEnv()
    evaded_count = 0
    results = []

    for i in range(n_flows):
        flow_evaded = False
        best_prob = 1.0

        for attempt in range(attempts_per_flow):
            state, _ = env.reset(seed=seed + i * attempts_per_flow + attempt)
            # Override with the same flow for all attempts of this flow
            if attempt == 0:
                original_flow = env.current_flow.copy()
            else:
                env.current_flow = original_flow.copy()
                state = original_flow.astype(np.float32)

            done = False
            while not done:
                action = env.action_space.sample()
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            prob = info.get('prob_malicious', 1.0)
            if prob < best_prob:
                best_prob = prob
            if info.get('evaded', False):
                flow_evaded = True
                break

        results.append({'evaded': flow_evaded, 'best_prob': best_prob})
        if flow_evaded:
            evaded_count += 1

        if (i + 1) % 200 == 0:
            print(f'  Random baseline: {i+1}/{n_flows}, evaded: {evaded_count}')

    rate = evaded_count / n_flows * 100
    return {
        'evasion_rate': rate,
        'evaded_count': evaded_count,
        'total_flows': n_flows,
        'attempts_per_flow': attempts_per_flow,
        'results': results,
    }


def exhaustive_baseline(n_flows=500, max_steps=3, seed=0):
    """Try all 12^max_steps deterministic action sequences per flow.

    Uses exact nominal magnitudes (no stochastic noise). This gives the
    theoretical ceiling for what max_steps steps can achieve.

    Note: 3 steps = 1,728 sequences (feasible). 5 steps = 248,832 sequences
    (very slow with RF prediction). Default is 3.

    Returns:
        dict with evasion rate and optimal sequences
    """
    env = WesternOC2EvasionEnv(max_steps=max_steps)
    n_actions = len(ACTION_DEFS)
    all_sequences = list(product(range(n_actions), repeat=max_steps))
    print(f'  Exhaustive: testing {len(all_sequences)} sequences per flow')

    evaded_count = 0
    results = []

    for i in range(n_flows):
        state, _ = env.reset(seed=seed + i)
        original_flow = env.current_flow.copy()

        best_prob = 1.0
        best_seq = None
        flow_evaded = False

        for seq in all_sequences:
            # Reset to original flow
            env.current_flow = original_flow.copy()
            env.original_flow = original_flow.copy()
            env.steps_taken = 0

            valid = True
            for action in seq:
                modified = _apply_deterministic(env.current_flow, action)
                modified = env._recalculate_derived(modified)

                if not env._is_valid(modified):
                    valid = False
                    break
                env.current_flow = modified

            if not valid:
                continue

            proba = env.clf.predict_proba([env.current_flow])[0]
            prob = 1.0 - proba[BENIGN_CLASS]
            if prob < best_prob:
                best_prob = prob
                best_seq = seq
            if prob < 0.5:
                flow_evaded = True
                best_seq = seq
                best_prob = prob
                break  # found an evasion, no need to keep searching

        results.append({
            'evaded': flow_evaded,
            'best_prob': best_prob,
            'best_sequence': best_seq,
        })
        if flow_evaded:
            evaded_count += 1

        if (i + 1) % 100 == 0:
            print(f'  Exhaustive: {i+1}/{n_flows}, evaded: {evaded_count}')

    rate = evaded_count / n_flows * 100
    return {
        'evasion_rate': rate,
        'evaded_count': evaded_count,
        'total_flows': n_flows,
        'results': results,
    }


def _apply_deterministic(flow, action):
    """Apply action with exact nominal magnitude (no noise)."""
    f = flow.copy()
    feat_name, direction, magnitude, _ = ACTION_DEFS[action]

    feat_idx = I[feat_name]
    factor = 1.0 + direction * magnitude
    new_val = f[feat_idx] * factor

    if feat_name == 'total_fwd_packets':
        if direction < 0:
            new_val = max(1, int(np.floor(new_val)))
        else:
            new_val = max(1, int(np.ceil(new_val)))
    elif feat_name == 'total_len_fwd_packets':
        new_val = max(1, round(new_val))
    else:
        new_val = max(0, new_val)

    f[feat_idx] = new_val
    return f


if __name__ == '__main__':
    print('=' * 60)
    print('BASELINE EVALUATION — Western-OC2-Lab RF (99.41% accuracy)')
    print('=' * 60)

    # Random baseline — single attempt (fair RL comparison)
    print('\n--- Random Baseline (1 attempt per flow) ---')
    r1 = random_baseline(n_flows=1000, attempts_per_flow=1, seed=0)
    print(f'  Evasion rate: {r1["evasion_rate"]:.1f}%')

    # Exhaustive search (3 steps — 1,728 sequences)
    print('\n--- Exhaustive Search (deterministic, all 1728 sequences, 3 steps) ---')
    ex = exhaustive_baseline(n_flows=500, max_steps=3, seed=0)
    print(f'  Evasion rate: {ex["evasion_rate"]:.1f}%')

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Random (1 attempt):   {r1["evasion_rate"]:.1f}%  <- RL must beat this')
    print(f'  Exhaustive (3-step):  {ex["evasion_rate"]:.1f}%  <- theoretical ceiling')
