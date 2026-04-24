#!/usr/bin/env python3
"""Compare checkpoint performance from two runs on one shared environment.

This script intentionally has no CLI options.
It uses two hardcoded run folders and one shared environment config.
For every checkpoint in each run, it executes 100 test episodes and records:
- mean return
- min return
- max return

It then writes a line-graph figure in the scripts directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_agents.agents.common.factory import load_agent, load_environment


SCRIPTS_DIR = Path(__file__).resolve().parent

# Shared environment for both checkpoint groups.
ENV_CONFIG = SCRIPTS_DIR / "configs/IntersectionEnv/env.json"

# Shared agent architecture used for both checkpoint groups.
AGENT_CONFIG = SCRIPTS_DIR / "configs/IntersectionEnv/agents/DQNAgent/baseline.json"

# Two hardcoded checkpoint groups requested by user.
CHECKPOINT_GROUPS = [
    {
        "name": "baseline_test1",
        "run_dir": SCRIPTS_DIR / "out/IntersectionEnv/DQNAgent/baseline_test1",
    },
    {
        "name": "multiagent_baseline",
        "run_dir": SCRIPTS_DIR / "out/MultiAgentIntersectionEnv/DQNAgent/baseline_20260422-202914_954380",
    },
]

EPISODES_PER_CHECKPOINT = 10
OUTPUT_FIGURE = SCRIPTS_DIR / "checkpoint_comparison_same_env.png"


def checkpoint_token(checkpoint_path: Path) -> str:
    return checkpoint_path.stem.replace("checkpoint-", "", 1)


def checkpoint_sort_key(checkpoint_path: Path):
    token = checkpoint_token(checkpoint_path)
    if token.isdigit():
        return (0, int(token))
    if token == "best":
        return (1, 10**9)
    if token == "final":
        return (2, 10**9)
    return (3, token)


def discover_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = sorted(run_dir.glob("checkpoint-*.tar"), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return checkpoints


def scalar_reward(reward) -> float:
    if isinstance(reward, (tuple, list, np.ndarray)):
        return float(np.sum(reward))
    return float(reward)


def run_test_episodes(checkpoint_path: Path, episodes: int) -> np.ndarray:
    env = load_environment(str(ENV_CONFIG))
    agent = load_agent(str(AGENT_CONFIG), env)
    agent.load(str(checkpoint_path))
    try:
        agent.eval()
    except AttributeError:
        pass

    episode_returns = []
    for episode in range(episodes):
        reset_output = env.reset(seed=episode)
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            observation, _ = reset_output
        else:
            observation = reset_output

        terminated = False
        truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action_sequence = agent.plan(observation)
            if not action_sequence:
                raise RuntimeError("Agent returned an empty action sequence")
            action = action_sequence[0]

            step_output = env.step(action)
            if len(step_output) == 5:
                observation, reward, terminated, truncated, _ = step_output
            else:
                observation, reward, done, _ = step_output
                terminated, truncated = bool(done), False

            total_reward += scalar_reward(reward)

        episode_returns.append(total_reward)

    env.close()
    return np.array(episode_returns, dtype=np.float64)


def evaluate_group(group_name: str, run_dir: Path) -> list[dict]:
    checkpoints = discover_checkpoints(run_dir)
    results = []

    print(f"\nEvaluating group: {group_name}")
    print(f"Run directory: {run_dir}")
    print(f"Shared env config: {ENV_CONFIG}")
    print(f"Checkpoints found: {len(checkpoints)}")

    for checkpoint in checkpoints:
        token = checkpoint_token(checkpoint)
        print(f"  -> Testing checkpoint-{token}.tar for {EPISODES_PER_CHECKPOINT} episodes")
        returns = run_test_episodes(checkpoint, EPISODES_PER_CHECKPOINT)
        results.append(
            {
                "checkpoint": token,
                "mean": float(np.mean(returns)),
                "min": float(np.min(returns)),
                "max": float(np.max(returns)),
            }
        )

    return results


def plot_results(group_results: dict[str, list[dict]]) -> None:
    figure, axes = plt.subplots(len(group_results), 1, figsize=(13, 5 * len(group_results)), squeeze=False)

    for index, (group_name, series) in enumerate(group_results.items()):
        axis = axes[index, 0]
        x = np.arange(len(series), dtype=np.int64)
        labels = [item["checkpoint"] for item in series]
        mean_values = np.array([item["mean"] for item in series], dtype=np.float64)
        min_values = np.array([item["min"] for item in series], dtype=np.float64)
        max_values = np.array([item["max"] for item in series], dtype=np.float64)

        axis.plot(x, mean_values, marker="o", linewidth=2.2, label="mean")
        axis.plot(x, min_values, marker="x", linewidth=1.6, linestyle="--", label="min")
        axis.plot(x, max_values, marker="x", linewidth=1.6, linestyle="--", label="max")
        axis.fill_between(x, min_values, max_values, alpha=0.12)

        axis.set_title(f"{group_name} (shared env: {ENV_CONFIG.name}, episodes={EPISODES_PER_CHECKPOINT})")
        axis.set_xlabel("Checkpoint")
        axis.set_ylabel("Episode return")
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.grid(alpha=0.25)
        axis.legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=180)
    plt.close(figure)
    print(f"\nSaved plot to: {OUTPUT_FIGURE}")


def main() -> None:
    all_results = {}
    for group in CHECKPOINT_GROUPS:
        all_results[group["name"]] = evaluate_group(group["name"], group["run_dir"])

    plot_results(all_results)


if __name__ == "__main__":
    main()
