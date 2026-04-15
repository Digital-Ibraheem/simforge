"""
Plotting utilities for experiment results.

WHAT PLOTS WE GENERATE:
  1. Learning curves:  success rate vs timesteps, one line per variant/seed.
     Shows how fast each approach learns during training.

  2. Bar chart:  success rate per policy variant on each OOD eval config.
     Shows in-distribution vs out-of-distribution performance at a glance.

  3. Heatmap:   variant × OOD config grid, cells = success rate.
     The clearest way to see the DR robustness tradeoff.

MATPLOTLIB PRIMER (if you're unfamiliar):
  - fig, axes = plt.subplots(rows, cols) — creates a figure with subplots
  - ax.plot(x, y, label="name") — line plot
  - ax.bar(x, heights) — bar chart
  - ax.set_xlabel/ylabel/title — labels
  - fig.tight_layout() — prevent overlapping elements
  - fig.savefig(path) — save to file (PNG for web, PDF for papers)
  - plt.close(fig) — free memory (important in scripts that make many figures)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# Consistent colors for the three DR variants across all plots
VARIANT_COLORS = {
    "no_dr": "#2196F3",        # blue
    "moderate_dr": "#4CAF50",  # green
    "aggressive_dr": "#FF5722", # orange-red
    "ppo": "#9C27B0",          # purple (Phase 4)
    "sac_her": "#2196F3",      # same as no_dr (for algo comparison)
}

VARIANT_LABELS = {
    "no_dr": "No DR",
    "moderate_dr": "Moderate DR",
    "aggressive_dr": "Aggressive DR",
    "ppo": "PPO (dense reward)",
    "sac_her": "SAC+HER (sparse reward)",
}


def plot_learning_curves(
    runs: dict[str, list[dict]],
    save_path: str | Path,
    title: str = "Training Learning Curves",
    smooth_window: int = 10,
) -> None:
    """
    Plot success rate vs timesteps for multiple variants, each with multiple seeds.

    Args:
        runs: {variant_name: [{"timesteps": [...], "success_rate": [...]}, ...]}
              Each value is a list of dicts, one per seed.
              Example:
                  runs = {
                      "no_dr": [
                          {"timesteps": [0, 10000, ...], "success_rate": [0, 0.1, ...]},
                          {"timesteps": [0, 10000, ...], "success_rate": [0, 0.05, ...]},
                      ],
                      "moderate_dr": [...]
                  }
        save_path:     Where to save the figure.
        title:         Plot title.
        smooth_window: Rolling average window size to reduce noise in curves.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for variant, seed_runs in runs.items():
        color = VARIANT_COLORS.get(variant, "#666666")
        label = VARIANT_LABELS.get(variant, variant)

        # Assume all seeds have the same timestep schedule
        all_success = np.array([r["success_rate"] for r in seed_runs])

        # Mean and std across seeds
        mean = np.mean(all_success, axis=0)
        std = np.std(all_success, axis=0)

        # Smooth with rolling average to reduce noise
        if smooth_window > 1:
            mean = _rolling_mean(mean, smooth_window)
            std = _rolling_mean(std, smooth_window)

        timesteps = seed_runs[0]["timesteps"]

        ax.plot(timesteps, mean, label=label, color=color, linewidth=2)
        # Shaded band = ±1 std across seeds (shows variance between seeds)
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_ood_bar_chart(
    results: dict[str, dict[str, float]],
    save_path: str | Path,
    title: str = "In-Distribution vs OOD Success Rate",
) -> None:
    """
    Bar chart showing success rate per variant across multiple eval configs.

    Args:
        results: {variant_name: {eval_config_name: success_rate}}
                 Example:
                     results = {
                         "no_dr": {
                             "default": 0.82, "heavy_object": 0.21, "low_friction": 0.18, ...
                         },
                         "moderate_dr": {
                             "default": 0.75, "heavy_object": 0.63, ...
                         },
                     }
        save_path: Where to save.
        title:     Plot title.
    """
    variants = list(results.keys())
    # Get eval config names from the first variant
    eval_configs = list(next(iter(results.values())).keys())
    n_configs = len(eval_configs)
    n_variants = len(variants)

    # Group bars by eval_config, with one bar per variant per group
    x = np.arange(n_configs)
    width = 0.8 / n_variants  # total bar width divided across variants

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, variant in enumerate(variants):
        color = VARIANT_COLORS.get(variant, "#666666")
        label = VARIANT_LABELS.get(variant, variant)
        heights = [results[variant].get(cfg, 0.0) for cfg in eval_configs]
        # Offset each variant's bars within each group
        offset = (i - n_variants / 2 + 0.5) * width
        ax.bar(x + offset, heights, width, label=label, color=color, alpha=0.85, edgecolor="white")

    ax.set_xlabel("Evaluation Config", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(eval_configs, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Add a vertical line separating "default" (in-dist) from OOD configs
    if "default" in eval_configs:
        ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.4, linewidth=1)
        ax.text(0.25, 1.02, "in-dist", ha="center", fontsize=9, transform=ax.get_xaxis_transform())
        ax.text(2.5, 1.02, "out-of-distribution →", ha="center", fontsize=9, transform=ax.get_xaxis_transform())

    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_heatmap(
    results: dict[str, dict[str, float]],
    save_path: str | Path,
    title: str = "Success Rate: Variant × Eval Config",
) -> None:
    """
    Heatmap with variants as rows and eval configs as columns.

    This is the most compact view of the ablation results — you can immediately
    see which variant handles which OOD condition.

    Args:
        results: Same format as plot_ood_bar_chart().
        save_path: Where to save.
        title: Plot title.
    """
    variants = list(results.keys())
    eval_configs = list(next(iter(results.values())).keys())

    # Build the 2D array: rows=variants, cols=eval_configs
    data = np.array([
        [results[v].get(cfg, 0.0) for cfg in eval_configs]
        for v in variants
    ])

    fig, ax = plt.subplots(figsize=(max(8, len(eval_configs) * 1.5), max(4, len(variants) * 0.8)))

    # imshow: viridis colormap goes from dark (low) to bright (high)
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")

    # Annotate each cell with the numeric value
    for i in range(len(variants)):
        for j in range(len(eval_configs)):
            val = data[i, j]
            # Use white text on dark cells, black on bright — pick based on value
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(eval_configs)))
    ax.set_xticklabels(eval_configs, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([VARIANT_LABELS.get(v, v) for v in variants], fontsize=10)
    ax.set_title(title, fontsize=14, pad=15)

    fig.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)
    fig.tight_layout()
    _save_figure(fig, save_path)


# --- Helpers ---

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Apply a rolling/moving average to smooth noisy curves."""
    out = np.convolve(arr, np.ones(window) / window, mode="valid")
    # Pad beginning so output length matches input
    pad = np.full(len(arr) - len(out), out[0])
    return np.concatenate([pad, out])


def _save_figure(fig: plt.Figure, path: str | Path) -> None:
    """Save figure as both PNG (for web/README) and PDF (for papers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # PNG: rasterized, good for GitHub README and W&B
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")

    # PDF: vector format, scales perfectly for paper figures
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")

    print(f"  Saved: {path.with_suffix('.png')} and {path.with_suffix('.pdf')}")
    plt.close(fig)
