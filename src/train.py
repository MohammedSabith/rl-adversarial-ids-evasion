"""
Train a PPO agent to evade the Western-OC2-Lab RF classifier.

Uses Stable-Baselines3 PPO with DummyVecEnv (faster than SubprocVecEnv for cheap envs).
"""

import os
import sys
from collections import Counter
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from src.western_oc2_environment import WesternOC2EvasionEnv, ACTION_DEFS


class EvasionMetricsCallback(BaseCallback):
    """Track evasion rate during training."""

    def __init__(self, max_steps=5, eval_freq=25_000, n_eval_flows=50, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_flows = n_eval_flows
        self.eval_env = WesternOC2EvasionEnv(max_steps=max_steps)
        self.results = []

    def _on_step(self):
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            evasion_rate, avg_reward = self._evaluate()
            self.results.append({
                'timestep': self.num_timesteps,
                'evasion_rate': evasion_rate,
                'avg_reward': avg_reward,
            })
            if self.verbose:
                print(f'  Step {self.num_timesteps:>7d}: '
                      f'evasion={evasion_rate:.1f}%, '
                      f'avg_reward={avg_reward:.3f}', flush=True)
        return True

    def _evaluate(self):
        evaded = 0
        total_rewards = []

        for i in range(self.n_eval_flows):
            state, _ = self.eval_env.reset(seed=10000 + i)
            ep_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(state, deterministic=True)
                state, reward, terminated, truncated, info = self.eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated

            total_rewards.append(ep_reward)
            if info.get('evaded', False):
                evaded += 1

        return evaded / self.n_eval_flows * 100, np.mean(total_rewards)


def make_env(max_steps, seed):
    def _init():
        env = WesternOC2EvasionEnv(max_steps=max_steps)
        env.reset(seed=seed)
        return env
    return _init


def train(max_steps=5, total_timesteps=100_000, seed=0, n_envs=4):
    """Train PPO agent and return the trained model + metrics."""

    env = DummyVecEnv([make_env(max_steps, seed + i) for i in range(n_envs)])

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=seed,
        verbose=0,
    )

    callback = EvasionMetricsCallback(
        max_steps=max_steps,
        eval_freq=25_000, n_eval_flows=50
    )

    print(f'Training PPO vs Western-OC2-Lab RF for {total_timesteps:,} timesteps (seed={seed})...')
    print(f'  Network: 78 -> 128 -> 128 -> 12')
    print(f'  Envs: {n_envs} (DummyVecEnv)')
    print(f'  Max steps: {max_steps}')
    print(flush=True)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model, callback.results


def evaluate(model, max_steps=5, n_flows=1000, seed=99999):
    """Final evaluation of a trained model."""
    env = WesternOC2EvasionEnv(max_steps=max_steps)
    evaded = 0
    rewards = []
    action_counts = np.zeros(len(ACTION_DEFS), dtype=int)
    step_actions = {s: [] for s in range(1, max_steps + 1)}

    for i in range(n_flows):
        state, _ = env.reset(seed=seed + i)
        ep_reward = 0
        done = False
        step = 0

        while not done:
            step += 1
            action, _ = model.predict(state, deterministic=True)
            action_counts[action] += 1
            step_actions[step].append(int(action))
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        if info.get('evaded', False):
            evaded += 1

    rate = evaded / n_flows * 100
    return {
        'evasion_rate': rate,
        'evaded_count': evaded,
        'total_flows': n_flows,
        'avg_reward': np.mean(rewards),
        'action_counts': action_counts,
        'step_actions': step_actions,
    }


if __name__ == '__main__':
    max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models', 'western_oc2')

    print('=' * 60)
    print('PPO TRAINING vs Western-OC2-Lab RF — Phase 1 Evasion Agent')
    print('=' * 60)

    model, training_metrics = train(
        max_steps=max_steps,
        total_timesteps=100_000,
        seed=0,
    )

    # Save
    model_path = os.path.join(models_dir, 'ppo_evasion')
    model.save(model_path)
    print(f'\nModel saved to {model_path}')

    metrics_path = os.path.join(models_dir, 'training_metrics.npy')
    np.save(metrics_path, training_metrics)

    # Evaluate
    print('\n' + '=' * 60)
    print(f'FINAL EVALUATION (1000 flows, {max_steps} steps)')
    print('=' * 60)

    result = evaluate(model, max_steps=max_steps, n_flows=1000)
    print(f'  Evasion rate: {result["evasion_rate"]:.1f}%')
    print(f'  Avg reward:   {result["avg_reward"]:.3f}')

    print('\n  Action usage (overall):')
    for i, count in enumerate(result['action_counts']):
        adef = ACTION_DEFS[i]
        sign = '+' if adef[1] > 0 else '-'
        name = f'{adef[0]} {sign}{int(adef[2]*100)}%'
        total = result['action_counts'].sum()
        pct = count / total * 100 if total > 0 else 0
        bar = '#' * int(pct)
        print(f'    {name:30s}: {count:5d} ({pct:4.1f}%) {bar}')

    print('\n  Per-step action distribution:')
    for step in range(1, max_steps + 1):
        acts = result['step_actions'][step]
        if not acts:
            continue
        counts = Counter(acts)
        top3 = counts.most_common(3)
        top_str = ', '.join(
            f'{ACTION_DEFS[a][0][:7]}{"+" if ACTION_DEFS[a][1]>0 else "-"}{int(ACTION_DEFS[a][2]*100)}%'
            for a, _ in top3
        )
        print(f'    Step {step}: top actions = {top_str}')
