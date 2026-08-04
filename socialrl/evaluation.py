from typing import Any, Dict, List

import numpy as np


def evaluate_policy(env: Any, agent: Any, episodes: int) -> List[Dict[str, float]]:
    """Evaluate a frozen policy without exploration or learning updates."""
    logs: List[Dict[str, float]] = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        rewards = 0.0
        successes = 0
        asocial_hits = 0
        observe_steps = 0
        detections = 0
        latencies = []

        while not done:
            action = agent.act_deterministic(state)
            state, reward, done, info = env.step(action)
            rewards += float(reward)
            observe_steps += int(info["observe"])
            detections += int(info["seen_lick"])
            asocial_hits += int(info["asocial_window_hit"])
            if info["eat_valid"]:
                successes += 1
                if info["latency_steps"] >= 0:
                    latencies.append(int(info["latency_steps"]))

        n_bouts = max(1, int(env.n_bouts))
        logs.append({
            "eval_episode": ep,
            "bout_success_rate": successes / n_bouts,
            "asocial_window_hit_rate": asocial_hits / n_bouts,
            "mean_reward": rewards,
            "obs_rate": observe_steps / max(1, env.max_steps),
            "obs_detect_rate": detections / max(1, observe_steps),
            "mean_latency_s": float(np.mean(latencies) * env.dt_s) if latencies else np.nan,
        })
    return logs
