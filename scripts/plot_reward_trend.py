#!/usr/bin/env python3
"""Plot episode reward trend from rl-agents evaluation logs.

This script reads lines such as:
    [rl_agents.trainer.evaluation:INFO] Episode 42 score: 3.7

and draws a reward curve across episodes, with optional moving average and
linear trend line.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPISODE_SCORE_PATTERN = re.compile(
    r"Episode\s+(?P<episode>\d+)\s+score:\s+(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def resolve_log_file(path: Path) -> Path:
    """Resolve an input path into a single logging file path.

    If `path` is a directory, search for logging files and use the most recent one.
    """
    if path.is_file():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    direct_logs = list(path.glob("logging*.log"))
    recursive_logs = list(path.rglob("logging*.log")) if not direct_logs else []
    candidates = direct_logs or recursive_logs
    if not candidates:
        raise FileNotFoundError(
            f"No logging file found under directory: {path}. "
            "Expected files like logging.<id>.<pid>.log"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_episode_rewards(log_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract episode indices and total rewards from a logging file."""
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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute a trailing moving average using a fixed-size window."""
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def plot_reward_trend(
    episodes: np.ndarray,
    rewards: np.ndarray,
    window: int,
    title: str,
    output: Path | None,
    show: bool,
) -> None:
    """Plot rewards and trend lines."""
    plt.figure(figsize=(11, 5.5))
    plt.plot(episodes, rewards, color="#4C78A8", alpha=0.35, linewidth=1.2, label="Episode reward")

    if len(rewards) >= window and window > 1:
        ma_values = moving_average(rewards, window)
        ma_episodes = episodes[window - 1 :]
        plt.plot(
            ma_episodes,
            ma_values,
            color="#F58518",
            linewidth=2.2,
            label=f"{window}-episode moving average",
        )

    if len(rewards) >= 2:
        slope, intercept = np.polyfit(episodes.astype(np.float64), rewards, 1)
        trend = slope * episodes + intercept
        plt.plot(episodes, trend, color="#54A24B", linestyle="--", linewidth=2.0, label="Linear trend")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=160)
        print(f"Saved plot to: {output}")

    if show:
        plt.show()
    else:
        plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read per-episode total reward from rl-agents logs and plot trend lines."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a logging file or a run directory containing logging*.log",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Moving-average window size (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (for example: reward_trend.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Episode Total Reward Trend",
        help="Plot title",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.window < 1:
        raise ValueError("--window must be >= 1")

    log_file = resolve_log_file(args.input_path)
    episodes, rewards = parse_episode_rewards(log_file)

    print(f"Using log file: {log_file}")
    print(f"Episodes parsed: {len(episodes)}")
    print(f"Reward min/mean/max: {rewards.min():.3f} / {rewards.mean():.3f} / {rewards.max():.3f}")

    plot_reward_trend(
        episodes=episodes,
        rewards=rewards,
        window=args.window,
        title=args.title,
        output=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
