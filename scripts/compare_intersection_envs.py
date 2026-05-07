#!/usr/bin/env python3
"""Run and compare the IntersectionEnv variants listed in the prompt.

The script executes each environment configuration with the matching DQN agent
config, then parses the generated logging files to build comparison plots. The
main multi-run graphs are rendered through ``plot_more_graphs.py`` and the
script also writes a summary bar chart with overall and tail-episode
performance.

Outputs are written in the scripts directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_SCRIPT = SCRIPT_DIR / "experiments.py"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from plot_more_graphs import graph_reward_curve, graph_histogram, graph_stability  # noqa: E402

AGENT_CONFIGS = {
    "baseline": SCRIPT_DIR / "configs/IntersectionEnv/agents/DQNAgent/baseline.json",
    "ego": SCRIPT_DIR / "configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json",
    "grid": SCRIPT_DIR / "configs/IntersectionEnv/agents/DQNAgent/grid_convnet.json",
}

RUNS = [
    # {
    #     "label": "baseline / env",
    #     "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env.json",
    #     "agent_config": AGENT_CONFIGS["baseline"],
    # },
    # {
    #     "label": "grid / grid_env",
    #     "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_grid_dense.json",
    #     "agent_config": AGENT_CONFIGS["baseline"],
    # },
    # {
    #     "label": "ego / env",
    #     "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env.json",
    #     "agent_config": AGENT_CONFIGS["baseline"],
    # },
    # {
    #     "label": "baseline / multi_model",
    #     "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_multi_agent.json",
    #     "agent_config": AGENT_CONFIGS["baseline"],
    # },
    {
        "label": "baseline / multi_agent1",
        "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_multi_agent1.json",
        "agent_config": AGENT_CONFIGS["baseline"],
    },
    # {
    #     "label": "baseline / multi_model2",
    #     "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_multi_agent2.json",
    #     "agent_config": AGENT_CONFIGS["baseline"],
    # },
    
    # {
    #     "label": "ego / multi_agent",
    #     "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_multi_agent.json",
    #     "agent_config": AGENT_CONFIGS["ego"],
    # },
    {
        "label": "ego / multi_agent1",
        "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_multi_agent1.json",
        "agent_config": AGENT_CONFIGS["ego"],
    },
    # {
    #     "label": "ego / multi_agent2",
    #     "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_multi_agent2.json",
    #     "agent_config": AGENT_CONFIGS["ego"],
    # },
    # {
    #     "label": "grid / multi_agent_grid",
    #     "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_multi_agent_grid.json",
    #     "agent_config": AGENT_CONFIGS["grid"],
    # },
    {
        "label": "grid / multi_agent_grid1",
        "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_multi_agent_grid1.json",
        "agent_config": AGENT_CONFIGS["grid"],
    },
    # {
    #     "label": "grid / multi_agent_grid2",
    #     "env_config": SCRIPT_DIR / "configs/IntersectionEnv/env_multi_agent_grid2.json",
    #     "agent_config": AGENT_CONFIGS["grid"],
    # },
]
DEFAULT_EPISODES = 10
DEFAULT_MOVING_AVERAGE_WINDOW = 2
DEFAULT_TAIL_WINDOW = 2
DEFAULT_OUTPUT_PREFIX = "intersection_env_comparison"

EPISODE_SCORE_PATTERN = re.compile(
    r"Episode\s+(?P<episode>\d+)\s+score:\s+(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass
class RunData:
    label: str
    env_config: Path
    agent_config: Path
    run_directory: Path
    log_file: Path
    episodes: np.ndarray
    rewards: np.ndarray
    average_episodes: np.ndarray
    average_rewards: np.ndarray
    overall_mean: float
    overall_std: float
    tail_mean: float
    tail_std: float
    best_episode_reward: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and compare IntersectionEnv variants.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--window", type=int, default=DEFAULT_MOVING_AVERAGE_WINDOW)
    parser.add_argument("--tail-window", type=int, default=DEFAULT_TAIL_WINDOW)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for the generated figure and JSON files.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip training and only rebuild plots from existing logs.",
    )
    return parser


def validate_inputs(episodes: int, window: int, tail_window: int) -> None:
    if episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if window < 1:
        raise ValueError("--window must be >= 1")
    if tail_window < 1:
        raise ValueError("--tail-window must be >= 1")


def find_latest_run_directory(base_directory: Path, previous_runs: set[Path]) -> Path:
    run_directories = sorted(
        [path for path in base_directory.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    new_runs = [path for path in run_directories if path not in previous_runs]
    if new_runs:
        return new_runs[-1]
    if run_directories:
        return run_directories[-1]
    raise FileNotFoundError(f"No run directory found under {base_directory}")


def locate_log_file(run_directory: Path) -> Path:
    log_files = sorted(run_directory.glob("logging*.log"), key=lambda path: path.stat().st_mtime)
    if not log_files:
        raise FileNotFoundError(f"No logging file found in {run_directory}")
    return log_files[-1]


def parse_rewards(log_file: Path) -> tuple[np.ndarray, np.ndarray]:
    rewards_by_episode: dict[int, float] = {}
    with log_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = EPISODE_SCORE_PATTERN.search(line)
            if match is None:
                continue
            episode = int(match.group("episode"))
            score = float(match.group("score"))
            rewards_by_episode[episode] = score

    if not rewards_by_episode:
        raise ValueError(
            f"No episode score lines found in {log_file}. Expected lines like 'Episode <n> score: <value>'."
        )

    episodes = np.array(sorted(rewards_by_episode.keys()), dtype=np.int64)
    rewards = np.array([rewards_by_episode[episode] for episode in episodes], dtype=np.float64)
    return episodes, rewards


def moving_average(episodes: np.ndarray, rewards: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1 or len(rewards) < window:
        return episodes.copy(), rewards.copy()

    kernel = np.ones(window, dtype=np.float64) / float(window)
    averaged_rewards = np.convolve(rewards, kernel, mode="valid")
    averaged_episodes = episodes[window - 1 :]
    return averaged_episodes, averaged_rewards


def summarize_series(rewards: np.ndarray, tail_window: int) -> tuple[float, float, float, float, float]:
    overall_mean = float(np.mean(rewards))
    overall_std = float(np.std(rewards))
    tail_slice = rewards[-min(tail_window, len(rewards)) :]
    tail_mean = float(np.mean(tail_slice))
    tail_std = float(np.std(tail_slice))
    best_reward = float(np.max(rewards))
    return overall_mean, overall_std, tail_mean, tail_std, best_reward


def run_environment(label: str, env_config: Path, agent_config: Path, episodes: int, seed: int | None) -> Path:
    command = [
        sys.executable,
        str(EXPERIMENTS_SCRIPT),
        "evaluate",
        str(env_config),
        str(agent_config),
        "--train",
        f"--episodes={episodes}",
        "--name-from-config",
        "--no-display",
    ]
    if seed is not None:
        command.append(f"--seed={seed}")

    print(f"\nRunning {label}")
    print(" ".join(command))
    base_out = SCRIPT_DIR / "out"
    base_out.mkdir(parents=True, exist_ok=True)
    previous_run_dirs = {p for p in base_out.rglob("*") if p.is_dir()}
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)
    new_dirs = [p for p in base_out.rglob("*") if p.is_dir() and p not in previous_run_dirs]
    dirs_with_logs = [p for p in new_dirs if list(p.glob("logging*.log"))]
    candidates = dirs_with_logs or new_dirs
    if not candidates:
        raise FileNotFoundError(f"No new run directory found under {base_out} after running {label}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def collect_run_data(
    label: str,
    env_config: Path,
    agent_config: Path,
    run_directory: Path,
    tail_window: int,
    window: int,
) -> RunData:
    log_file = locate_log_file(run_directory)
    episodes, rewards = parse_rewards(log_file)
    average_episodes, average_rewards = moving_average(episodes, rewards, window)
    overall_mean, overall_std, tail_mean, tail_std, best_reward = summarize_series(rewards, tail_window)

    return RunData(
        label=label,
        env_config=env_config,
        agent_config=agent_config,
        run_directory=run_directory,
        log_file=log_file,
        episodes=episodes,
        rewards=rewards,
        average_episodes=average_episodes,
        average_rewards=average_rewards,
        overall_mean=overall_mean,
        overall_std=overall_std,
        tail_mean=tail_mean,
        tail_std=tail_std,
        best_episode_reward=best_reward,
    )


def render_additional_graphs(
    collected_runs: list[RunData],
    output_prefix: str,
    window: int,
    tail_window: int,
) -> Path:
    graph_output_dir = SCRIPT_DIR / f"{output_prefix}_graphs"
    graph_output_dir.mkdir(parents=True, exist_ok=True)
    helper_window = max(window * 2, tail_window)
    runs = [{"label": rd.label, "episodes": rd.episodes, "rewards": rd.rewards} for rd in collected_runs]

    print("\nRendering comparison graphs")
    graph_reward_curve(runs, graph_output_dir, window, helper_window)
    graph_histogram(runs, graph_output_dir)
    graph_stability(runs, graph_output_dir, tail_window)
    return graph_output_dir


def plot_summary_bars(runs: list[RunData], output_path: Path, tail_window: int) -> None:
    labels = [run.label for run in runs]
    overall_means = np.array([run.overall_mean for run in runs], dtype=np.float64)
    tail_means = np.array([run.tail_mean for run in runs], dtype=np.float64)
    overall_stds = np.array([run.overall_std for run in runs], dtype=np.float64)
    tail_stds = np.array([run.tail_std for run in runs], dtype=np.float64)

    x_positions = np.arange(len(runs), dtype=np.float64)
    bar_width = 0.38

    figure, axis = plt.subplots(figsize=(13.5, 6.5))
    axis.bar(
        x_positions - bar_width / 2,
        overall_means,
        width=bar_width,
        yerr=overall_stds,
        capsize=4,
        label="overall mean",
    )
    axis.bar(
        x_positions + bar_width / 2,
        tail_means,
        width=bar_width,
        yerr=tail_stds,
        capsize=4,
        label=f"last {tail_window} mean",
    )

    axis.set_title("IntersectionEnv comparison - aggregate reward summary")
    axis.set_ylabel("Episode reward")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_summary_json(runs: list[RunData], output_path: Path) -> None:
    payload = {
        run.label: {
            "env_config": str(run.env_config),
            "agent_config": str(run.agent_config),
            "run_directory": str(run.run_directory),
            "log_file": str(run.log_file),
            "episodes": int(len(run.episodes)),
            "overall_mean": run.overall_mean,
            "overall_std": run.overall_std,
            "tail_mean": run.tail_mean,
            "tail_std": run.tail_std,
            "best_episode_reward": run.best_episode_reward,
        }
        for run in runs
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    validate_inputs(args.episodes, args.window, args.tail_window)

    for run in RUNS:
        if not run["env_config"].exists():
            raise FileNotFoundError(f"Missing environment config: {run['env_config']}")
        if not run["agent_config"].exists():
            raise FileNotFoundError(f"Missing agent config: {run['agent_config']}")

    collected_runs: list[RunData] = []
    if args.skip_run:
        summary_path = SCRIPT_DIR / f"{args.output_prefix}_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"--skip-run was set, but no summary file was found at {summary_path}"
            )
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
        for run in RUNS:
            run_summary = summary.get(run["label"])
            if run_summary is None:
                raise KeyError(f"Missing run '{run['label']}' in summary file {summary_path}")
            collected_runs.append(
                collect_run_data(
                    run["label"],
                    run["env_config"],
                    run["agent_config"],
                    Path(run_summary["run_directory"]),
                    args.tail_window,
                    args.window,
                )
            )
    else:
        for run in RUNS:
            run_directory = run_environment(run["label"], run["env_config"], run["agent_config"], args.episodes, args.seed)
            collected_runs.append(
                collect_run_data(
                    run["label"],
                    run["env_config"],
                    run["agent_config"],
                    run_directory,
                    args.tail_window,
                    args.window,
                )
            )

    summary_output = SCRIPT_DIR / f"{args.output_prefix}_summary.png"
    json_output = SCRIPT_DIR / f"{args.output_prefix}_summary.json"
    graph_output_dir = render_additional_graphs(
        collected_runs,
        args.output_prefix,
        args.window,
        args.tail_window,
    )

    plot_summary_bars(collected_runs, summary_output, args.tail_window)
    save_summary_json(collected_runs, json_output)

    print(f"\nSaved comparison graphs to: {graph_output_dir}")
    print(f"Saved summary figure to: {summary_output}")
    print(f"Saved metrics summary to: {json_output}")


if __name__ == "__main__":
    main()