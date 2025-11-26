from typing import Dict

import flax.linen as nn
import gymnasium as gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            observation, _ = reset_out
        else:
            observation = reset_out
        done = False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            result = env.step(action)
            if len(result) == 5:
                observation, _, terminated, truncated, info = result
                done = terminated or truncated
            else:
                observation, _, done, info = result

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
