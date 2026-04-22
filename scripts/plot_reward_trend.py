#!/usr/bin/env python3
"""Plot reward trend from one or more run folders.

For each folder, this script finds the most recent logging*.log file, extracts
episode scores from lines such as:
    [rl_agents.trainer.evaluation:INFO] Episode 42 score: 3.7

Then it creates one image in the current working directory:
1) trend.png
    - Raw moving-average reward curves with per-run linear trends.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
from pathlib import Path


EPISODE_SCORE_PATTERN = re.compile(
    r"Episode\s+(?P<episode>\d+)\s+score:\s+(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def import_plot_dependencies(no_show: bool):
    """Import plotting dependencies and configure headless backend when needed."""
    try:
        matplotlib = importlib.import_module("matplotlib")
        if no_show and "MPLBACKEND" not in os.environ:
            matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        np = importlib.import_module("numpy")
    except ModuleNotFoundError as exc:
        package_name = getattr(exc, "name", "a required package")
        raise ModuleNotFoundError(
            f"Missing dependency: {package_name}. Install required packages with: "
            "pip install numpy matplotlib"
        ) from exc

    return np, plt


def resolve_log_file(folder: Path) -> Path:
    """Find the most recent logging file in a run folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected a folder path, got file: {folder}")

    direct_logs = list(folder.glob("logging*.log"))
    recursive_logs = list(folder.rglob("logging*.log")) if not direct_logs else []
    candidates = direct_logs or recursive_logs
    if not candidates:
        raise FileNotFoundError(
            f"No logging file found under folder: {folder}. "
            "Expected files like logging.<id>.<pid>.log"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_episode_rewards(log_file: Path, np):
    """Extract episode indices and rewards from a logging file."""
    rewards_by_episode: dict[int, float] = {}

    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            match = EPISODE_SCORE_PATTERN.search(line)
            if not match:
                continue
            episode = int(match.group("episode"))
            score = float(match.group("score"))
            rewards_by_episode[episode] = score

    if not rewards_by_episode:
        raise ValueError(
            f"No episode reward lines were found in {log_file}. "
            "Expected lines containing 'Episode <n> score: <value>'."
        )

    episodes = np.array(sorted(rewards_by_episode.keys()), dtype=np.int64)
    rewards = np.array([rewards_by_episode[e] for e in episodes], dtype=np.float64)
    return episodes, rewards


def moving_average(episodes, rewards, window: int, np):
    """Compute trailing moving average and aligned episode indices."""
    if window <= 1 or len(rewards) < window:
        return episodes, rewards.copy()

    kernel = np.ones(window, dtype=np.float64) / float(window)
    avg_rewards = np.convolve(rewards, kernel, mode="valid")
    avg_episodes = episodes[window - 1 :]
    return avg_episodes, avg_rewards


def collect_run_data(folder: Path, window: int, np):
    """Read one folder and return parsed reward series."""
    log_file = resolve_log_file(folder)
    episodes, rewards = parse_episode_rewards(log_file, np)
    avg_episodes, avg_rewards = moving_average(episodes, rewards, window, np)
    label = folder.name if folder.name else str(folder)

    return {
        "folder": folder,
        "label": label,
        "log_file": log_file,
        "episodes": episodes,
        "rewards": rewards,
        "avg_episodes": avg_episodes,
        "avg_rewards": avg_rewards,
    }


def plot_runs(
    runs,
    title: str,
    output_name: str,
    np,
    plt,
):
    """Plot raw moving-average reward trends for multiple runs."""
    plt.figure(figsize=(11, 5.5))
    plotted_values = []
    plotted_episodes = []

    for run in runs:
        x = run["avg_episodes"]
        y = run["avg_rewards"]
        plotted_values.append(y)
        plotted_episodes.append(x)
        line = plt.plot(x, y, linewidth=2.0, label=run["label"])[0]

        # Add a linear trend line for each run.
        if len(y) >= 2:
            slope, intercept = np.polyfit(x.astype(np.float64), y, 1)
            trend = slope * x + intercept
            plotted_values.append(trend)
            plt.plot(
                x,
                trend,
                linestyle="--",
                linewidth=1.6,
                alpha=0.85,
                color=line.get_color(),
                label=f"{run['label']} trend",
            )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    if plotted_values:
        y_min = min(float(np.min(arr)) for arr in plotted_values if len(arr) > 0)
        y_max = max(float(np.max(arr)) for arr in plotted_values if len(arr) > 0)
        if y_min == y_max:
            pad = 1e-6 if y_min == 0 else abs(y_min) * 1e-6
            y_min -= pad
            y_max += pad
        plt.ylim(y_min, y_max)
    if plotted_episodes:
        x_min = min(float(np.min(arr)) for arr in plotted_episodes if len(arr) > 0)
        x_max = max(float(np.max(arr)) for arr in plotted_episodes if len(arr) > 0)
        if x_min == x_max:
            x_max += 1.0
        plt.xlim(x_min, x_max)
    plt.margins(x=0, y=0)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = Path.cwd() / output_name
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved plot to: {output_path}")


def summarize_run(run):
    rewards = run["rewards"]
    print(f"Using log file: {run['log_file']}")
    print(f"Episodes parsed: {len(run['episodes'])}")
    print(f"Reward min/mean/max: {rewards.min():.3f} / {rewards.mean():.3f} / {rewards.max():.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read per-episode rewards from multiple run folders and generate "
            "one raw reward trend image in the current working directory."
        )
    )
    parser.add_argument(
        "folders",
        nargs="+",
        type=Path,
        help="One or more run folders (each containing logging*.log)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Moving-average window size (default: 50)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Deprecated flag kept for compatibility. Plots are always saved without showing.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.window < 1:
        raise ValueError("--window must be >= 1")

    np, plt = import_plot_dependencies(no_show=True)

    runs = [collect_run_data(folder, args.window, np) for folder in args.folders]
    for run in runs:
        summarize_run(run)

    plot_runs(
        runs=runs,
        title=f"Reward Trend ({args.window}-episode average), moving averages only",
        output_name="trend.png",
        np=np,
        plt=plt,
    )


if __name__ == "__main__":
    main()
