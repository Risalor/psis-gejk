#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

EPISODE_SCORE_PATTERN = re.compile(
    r"Episode\s+(?P<episode>\d+)\s+score:\s+(?P<score>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)



def resolve_log_file(path: Path) -> Path:
    """Accept a .log file directly or find the newest logging*.log in a folder."""
    if path.is_file():
        return path
    if path.is_dir():
        candidates = list(path.glob("logging*.log")) or list(path.rglob("logging*.log"))
        if not candidates:
            raise FileNotFoundError(f"No logging*.log found under: {path}")
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"Path does not exist: {path}")


def parse_log(log_file: Path):
    """Return (episodes, rewards) arrays from a log file."""
    rewards_by_ep: dict[int, float] = {}
    with log_file.open(encoding="utf-8") as f:
        for line in f:
            m = EPISODE_SCORE_PATTERN.search(line)
            if m:
                rewards_by_ep[int(m.group("episode"))] = float(m.group("score"))
    if not rewards_by_ep:
        raise ValueError(f"No 'Episode N score: X' lines found in {log_file}")
    episodes = np.array(sorted(rewards_by_ep), dtype=np.int64)
    rewards  = np.array([rewards_by_ep[e] for e in episodes], dtype=np.float64)
    return episodes, rewards


def moving_average(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def load_runs(paths: list[Path]):
    runs = []
    for path in paths:
        log_file = resolve_log_file(path)
        episodes, rewards = parse_log(log_file)
        label = path.name if path.is_dir() else path.stem
        print(f"  {label}: {len(episodes)} episodes, "
              f"min={rewards.min():.2f} mean={rewards.mean():.2f} max={rewards.max():.2f}")
        runs.append({"label": label, "episodes": episodes, "rewards": rewards})
    return runs


def load_runs_with_labels(paths: list[Path], labels: list[str] | None):
    """Load all runs and optionally override the display labels."""
    runs = load_runs(paths)
    if labels is None:
        return runs
    if len(labels) != len(runs):
        raise ValueError(f"Expected {len(runs)} labels, got {len(labels)}")
    for run, label in zip(runs, labels):
        run["label"] = label
    return runs



def graph_reward_curve(runs, out_dir: Path, window1: int, window2: int):
    """Moving averages + linear trend lines for all runs."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for run in runs:
        ep, rw = run["episodes"], run["rewards"]
        label  = run["label"]
        color  = None

        if len(rw) >= window1:
            line, = ax.plot(ep[window1 - 1:], moving_average(rw, window1),
                            lw=1.8, label=f"{label} (MA {window1})")
            color = line.get_color()
        else:
            line, = ax.plot(ep, rw, lw=1.5, label=label)
            color = line.get_color()

        if len(rw) >= window2:
            ax.plot(ep[window2 - 1:], moving_average(rw, window2),
                    lw=2.5, ls="-", alpha=0.6, color=color,
                    label=f"{label} (MA {window2})")

        # trend line
        slope, intercept = np.polyfit(ep.astype(float), rw, 1)
        ax.plot(ep, slope * ep + intercept,
                lw=1.2, ls="--", alpha=0.75, color=color,
                label=f"{label} (trendline)")

    ax.set_title("Reward Curve — Moving Averages & Trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, borderaxespad=0.0)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    out = out_dir / "01_reward_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def graph_histogram(runs, out_dir: Path):
    """Overlapping score distribution histograms."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for run in runs:
        ax.hist(run["rewards"], bins=40, alpha=0.5, label=run["label"], edgecolor="white", linewidth=0.3)

    ax.set_title("Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, borderaxespad=0.0)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    out = out_dir / "02_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def graph_stability(runs, out_dir: Path, std_window: int):
    """Rolling std dev per run."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for run in runs:
        ep, rw = run["episodes"], run["rewards"]
        if len(rw) < std_window:
            print(f"  Skipping stability for '{run['label']}' — too few episodes for window {std_window}")
            continue
        std_vals = [rw[max(0, i - std_window): i].std() for i in range(std_window, len(rw) + 1)]
        ax.plot(ep[std_window - 1:], std_vals, lw=1.6, label=run["label"])

    ax.set_title(f"Rolling Std Dev ({std_window}-ep) — Lower = More Stable")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Std Dev of Score")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, borderaxespad=0.0)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    out = out_dir / "03_stability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def graph_comparison_bar(runs, out_dir: Path):
    labels = [run["label"] for run in runs]
    means = [run["rewards"].mean() for run in runs]
    bests = [run["rewards"].max() for run in runs]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, means, width, label='Average Reward', color='#3498db')
    rects2 = ax.bar(x + width/2, bests, width, label='Best Episode', color='#2ecc71')

    ax.set_ylabel('Reward Score')
    ax.set_title('Environment Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    out = out_dir / "04_env_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot RL training graphs from one or more log files or run folders."
    )
    p.add_argument(
        "paths", nargs="+", type=Path,
        help="One or more log files or folders containing logging*.log files.",
    )
    p.add_argument(
        "--labels", nargs="*", default=None,
        help="Optional display labels for the runs, in the same order as the paths.",
    )
    p.add_argument(
        "--window", type=int, default=50,
        help="Short moving-average window (default: 50)",
    )
    p.add_argument(
        "--window2", type=int, default=200,
        help="Long moving-average window (default: 200)",
    )
    p.add_argument(
        "--std-window", type=int, default=100,
        help="Window for rolling std-dev stability graph (default: 100)",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("graphs"),
        help="Output folder for graph PNGs (default: graphs/)",
    )
    return p


def main():
    args = build_parser().parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs_with_labels(args.paths, args.labels)

    print(":o make graph :0")
    graph_reward_curve(runs, args.out_dir, args.window, args.window2)
    graph_histogram(runs, args.out_dir)
    graph_stability(runs, args.out_dir, args.std_window)

    print(f"\nsaved to: {args.out_dir.resolve()}/")


if __name__ == "__main__":
    main()
