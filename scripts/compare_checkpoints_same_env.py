#!/usr/bin/env python3
"""Compare best/final checkpoint performance from multiple runs on one shared environment.

For each run only checkpoint-best.tar and checkpoint-final.tar are tested.
Each run gets its own PNG in envComparisonGraphs/.
A combined histogram PNG shows score distributions across all runs/checkpoints
based on 100 test episodes each.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agents.agents.common.factory import load_agent, load_environment


SCRIPTS_DIR = Path(__file__).resolve().parent

# ENV_CONFIG   = SCRIPTS_DIR / "configs/IntersectionEnv/env.json"
# AGENT_CONFIG = SCRIPTS_DIR / "configs/IntersectionEnv/agents/DQNAgent/baseline.json"

# CHECKPOINT_GROUPS = [
#     {
#         "name": "baseline_env",
#         "run_dir": SCRIPTS_DIR / "out/IntersectionEnv/DQNAgent/baseline_env",
#     },
#     {
#         "name": "multiagent_baseline_env0",
#         "run_dir": SCRIPTS_DIR / "out/MultiAgentIntersectionEnv/DQNAgent/baseline_multiagent_env0",
#     },
#     {
#         "name": "multiagent_baseline_env1",
#         "run_dir": SCRIPTS_DIR / "out/MultiAgentIntersectionEnv/DQNAgent/baseline_multiagent_env1",
#     },
#     {
#         "name": "multiagent_baseline_env2",
#         "run_dir": SCRIPTS_DIR / "out/MultiAgentIntersectionEnv/DQNAgent/baseline_multiagent_env2",
#     },
# ]
ENV_CONFIG   = SCRIPTS_DIR / "configs/IntersectionEnv/env.json"
AGENT_CONFIG = SCRIPTS_DIR / "configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json"

CHECKPOINT_GROUPS = [
    {
        "name": "ego_attention",
        "run_dir": SCRIPTS_DIR / "out/IntersectionEnv/DQNAgent/ego_attention_env",
    },
    {
        "name": "multiagent_ego_attention_env0",
        "run_dir": SCRIPTS_DIR / "out/MultiAgentIntersectionEnv/DQNAgent/ego_attention_multiagent_env0",
    },
    {
        "name": "multiagent_ego_attention_env1",
        "run_dir": SCRIPTS_DIR / "out/MultiAgentIntersectionEnv/DQNAgent/ego_attention_multiagent_env1",
    },
    {
        "name": "multiagent_ego_attention_env2",
        "run_dir": SCRIPTS_DIR / "out/MultiAgentIntersectionEnv/DQNAgent/ego_attention_multiagent_env2",
    },
]
# ENV_CONFIG   = SCRIPTS_DIR / "configs/IntersectionEnv/env_grid_dense.json"
# AGENT_CONFIG = SCRIPTS_DIR / "configs/IntersectionEnv/agents/DQNAgent/grid_convnet.json"

# CHECKPOINT_GROUPS = [
#     {
#         "name": "lobotomy",
#         "run_dir": SCRIPTS_DIR / "out/MultiAgentIntersectionEnv/DQNAgent/grid_convnet_multiagent_env0",
#     },
    
# ]
EPISODES_PER_CHECKPOINT = 100
OUTPUT_DIR = SCRIPTS_DIR / "envComparisonGraphs"


def discover_checkpoints(run_dir: Path) -> list[Path]:
    """Return [ checkpoint-final.tar] that exist in run_dir."""
    candidates = ["checkpoint-final.tar"]
    found = [run_dir / name for name in candidates if (run_dir / name).exists()]
    if not found:
        raise FileNotFoundError(f"No best/final checkpoints found in {run_dir}")
    return found


def checkpoint_label(checkpoint_path: Path) -> str:
    return checkpoint_path.stem.replace("checkpoint-", "", 1)


def scalar_reward(reward) -> float:
    if isinstance(reward, (tuple, list, np.ndarray)):
        return float(np.sum(reward))
    return float(reward)


def run_test_episodes(checkpoint_path: Path, episodes: int) -> np.ndarray:
    env   = load_environment(str(ENV_CONFIG))
    agent = load_agent(str(AGENT_CONFIG), env)
    agent.load(str(checkpoint_path))
    try:
        agent.eval()
    except AttributeError:
        pass

    episode_returns = []
    for episode in range(episodes):
        reset_output = env.reset(seed=episode)
        observation  = reset_output[0] if isinstance(reset_output, tuple) else reset_output

        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action_sequence = agent.plan(observation)
            if not action_sequence:
                raise RuntimeError("Agent returned an empty action sequence")
            step_output = env.step(action_sequence[0])
            if len(step_output) == 5:
                observation, reward, terminated, truncated, _ = step_output
            else:
                observation, reward, done, _ = step_output
                terminated, truncated = bool(done), False
            total_reward += scalar_reward(reward)

        episode_returns.append(total_reward)

    env.close()
    return np.array(episode_returns, dtype=np.float64)


def evaluate_group(name: str, run_dir: Path) -> list[dict]:
    checkpoints = discover_checkpoints(run_dir)
    results = []

    print(f"\nEvaluating: {name}  ({run_dir.name})")
    for ckpt in checkpoints:
        label = checkpoint_label(ckpt)
        print(f"  -> {ckpt.name}  ({EPISODES_PER_CHECKPOINT} episodes)")
        returns = run_test_episodes(ckpt, EPISODES_PER_CHECKPOINT)
        results.append({
            "checkpoint": label,
            "returns":    returns,
            "mean":       float(np.mean(returns)),
            "min":        float(np.min(returns)),
            "max":        float(np.max(returns)),
            "std":        float(np.std(returns)),
        })
        print(f"     mean={results[-1]['mean']:.3f}  std={results[-1]['std']:.3f}  "
              f"min={results[-1]['min']:.3f}  max={results[-1]['max']:.3f}")

    return results


def _histogram_axes(axes_row: list, series: list[dict], title_prefix: str) -> None:
    """Fill a row of axes (one per checkpoint) with individual histograms."""
    colors = ["steelblue", "tomato"]
    all_vals  = np.concatenate([item["returns"] for item in series])
    x_min     = np.floor(all_vals.min() * 2) / 2
    x_max     = np.ceil(all_vals.max() * 2) / 2
    bin_edges = np.arange(x_min - 0.25, x_max + 0.5, 0.5)
    ticks     = np.arange(x_min, x_max + 0.5, 0.5)

    for idx, (ax, item) in enumerate(zip(axes_row, series)):
        ax.hist(
            item["returns"],
            bins=bin_edges,
            color=colors[idx % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(f"{title_prefix}  —  {item['checkpoint']}  (n={EPISODES_PER_CHECKPOINT})")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(v)}" if v == int(v) and int(v) % 2 == 0 else "" for v in ticks])
        ax.grid(axis="y", alpha=0.25)


def plot_group(name: str, series: list[dict], out_dir: Path) -> None:
    """One subplot per checkpoint (best / final) side by side."""
    fig, axes = plt.subplots(1, len(series), figsize=(10, 5), squeeze=False)
    _histogram_axes(list(axes[0]), series, name)
    fig.tight_layout()
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_combined_histogram(all_groups: list[dict], out_dir: Path) -> None:
    """Grid: one row per agent, one column per checkpoint (best / final)."""
    n_groups = len(all_groups)
    n_cols   = max(len(g["results"]) for g in all_groups)
    fig, axes = plt.subplots(n_groups, n_cols, figsize=(10 * n_cols, 5 * n_groups), squeeze=False)

    for row, group in enumerate(all_groups):
        _histogram_axes(list(axes[row]), group["results"], group["name"])

    fig.tight_layout()
    out = out_dir / "all_histogram.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_groups = []
    for group in CHECKPOINT_GROUPS:
        results = evaluate_group(group["name"], group["run_dir"])
        all_groups.append({"name": group["name"], "results": results})

    print("\nPlotting individual group graphs...")
    for group in all_groups:
        plot_group(group["name"], group["results"], OUTPUT_DIR)

    print("Plotting combined histogram...")
    plot_combined_histogram(all_groups, OUTPUT_DIR)


if __name__ == "__main__":
    main()
