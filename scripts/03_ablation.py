"""
Phase 3: Ablation study — No DR vs Moderate DR vs Aggressive DR.

PURPOSE:
--------
An ablation study isolates the effect of a single variable (here: DR level)
by keeping everything else identical. We train 3 × 3 = 9 runs:

    Variant          Config                  Seeds
    --------         ------                  -----
    no_dr            dr_off.yaml             42, 1, 2
    moderate_dr      dr_moderate.yaml        42, 1, 2
    aggressive_dr    dr_aggressive.yaml      42, 1, 2

Then evaluate all 9 trained policies under 6 physics conditions (default + 5 OOD).

The key questions this answers:
  1. Does DR hurt in-distribution performance? (expected: yes, slightly)
  2. Does DR improve out-of-distribution performance? (expected: yes, significantly)
  3. Is more DR always better? (expected: no — aggressive DR may over-regularize)

WHY 3 SEEDS?
  RL training has high variance — the same hyperparameters with different random
  seeds can converge to quite different policies. 3 seeds is the minimum for
  seeing whether a difference is real or just random variation. Report mean ± std.

TRAINING TIME:
  9 runs × 1M steps each ≈ 27-45 hours on CPU.
  Run overnight or across multiple sessions. Checkpoints are saved every 50k steps.

USAGE:
  # Run all variants sequentially (takes ~2 days on CPU):
  python scripts/03_ablation.py

  # Run just one variant (then run the others separately):
  python scripts/03_ablation.py --variant no_dr
  python scripts/03_ablation.py --variant moderate_dr
  python scripts/03_ablation.py --variant aggressive_dr

  # Skip training (evaluate already-trained models):
  python scripts/03_ablation.py --eval-only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simforge.training.train import TrainingConfig, train

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# The three variants: name → DR config file
VARIANTS = {
    "no_dr": CONFIGS_DIR / "dr_off.yaml",
    "moderate_dr": CONFIGS_DIR / "dr_moderate.yaml",
    "aggressive_dr": CONFIGS_DIR / "dr_aggressive.yaml",
}

SEEDS = [42, 1, 2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run DR ablation study")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()) + ["all"],
        default="all",
        help="Which variant to train (default: all)",
    )
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, just run OOD evaluation on saved models",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.eval_only:
        # --- Train ---
        variants_to_run = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]
        for variant in variants_to_run:
            for seed in SEEDS:
                _train_one(variant, seed, args.timesteps, not args.no_wandb)

    # --- Evaluate all trained models on OOD configs ---
    print("\n\nRunning OOD evaluation...")
    results = _run_ood_eval()

    # --- Generate plots ---
    print("\nGenerating plots...")
    _generate_plots(results)


def _train_one(variant: str, seed: int, timesteps: int, use_wandb: bool):
    """Train one variant at one seed."""
    dr_config = str(VARIANTS[variant])
    run_name = f"ablation_{variant}_s{seed}"

    print(f"\n{'='*50}")
    print(f"Training: {run_name}")
    print(f"DR config: {dr_config}")
    print(f"{'='*50}")

    cfg = TrainingConfig(
        env_id="FetchPickAndPlace-v4",
        reward_type="sparse",
        algorithm="SAC",
        use_her=True,
        learning_rate=1e-3,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=1_000,
        gamma=0.95,
        tau=0.05,
        n_sampled_goal=4,
        goal_selection_strategy="future",
        total_timesteps=timesteps,
        eval_freq=10_000,
        n_eval_episodes=20,
        seed=seed,
        dr_config=dr_config,
        name=run_name,
        model_dir=str(RESULTS_DIR / "models" / "ablation"),
        log_dir=str(RESULTS_DIR / "logs" / "ablation"),
        video_dir=str(RESULTS_DIR / "videos" / "ablation"),
        use_wandb=use_wandb,
        wandb_project="simforge",
        wandb_group="phase3-ablation",
        wandb_tags=["ablation", variant, f"seed-{seed}"],
    )

    train(cfg)


def _run_ood_eval():
    """
    Evaluate all trained ablation models under OOD physics.

    Returns a nested dict:
        results[variant][ood_config_name] = {
            "success_rate": ...,
            "mean": ...,   (across seeds)
            "std": ...,
        }
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import yaml

    from simforge.evaluation.evaluate import evaluate_policy, evaluate_under_ood, load_model

    # Load OOD eval configs
    ood_config_path = CONFIGS_DIR / "eval_ood.yaml"
    with open(ood_config_path) as f:
        ood_cfg = yaml.safe_load(f)
    ood_configs = ood_cfg["eval_configs"]
    n_eval_episodes = ood_cfg["evaluation"]["n_episodes"]

    results = {}

    for variant in VARIANTS:
        results[variant] = {}
        seed_results = {cfg["name"]: [] for cfg in ood_configs}

        for seed in SEEDS:
            run_name = f"ablation_{variant}_s{seed}"
            model_path = RESULTS_DIR / "models" / "ablation" / run_name / "best" / "best_model"

            if not model_path.with_suffix(".zip").exists():
                print(f"  WARNING: model not found: {model_path} — skipping seed {seed}")
                continue

            print(f"\n  Evaluating {run_name}...")
            env = gym.make("FetchPickAndPlace-v4")
            model = load_model(model_path, env, algorithm="SAC")

            for ood_config in ood_configs:
                res = evaluate_under_ood(model, env, ood_config, n_episodes=n_eval_episodes)
                seed_results[ood_config["name"]].append(res["success_rate"])
                print(f"    {ood_config['name']:20s}: {res['success_rate']:.1%}")

            env.close()

        # Aggregate across seeds
        for cfg_name, seed_rates in seed_results.items():
            if seed_rates:
                results[variant][cfg_name] = float(np.mean(seed_rates))
            else:
                results[variant][cfg_name] = 0.0

    return results


def _generate_plots(results: dict):
    """Generate and save all ablation plots."""
    import numpy as np

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    from simforge.evaluation.plotting import plot_ood_bar_chart, plot_heatmap

    plot_ood_bar_chart(
        results,
        save_path=plots_dir / "ablation_ood_bar",
        title="Domain Randomization Ablation: In-Dist vs OOD Success Rate",
    )

    plot_heatmap(
        results,
        save_path=plots_dir / "ablation_heatmap",
        title="Ablation Heatmap: Success Rate by Variant and Eval Config",
    )

    print(f"\nPlots saved to {plots_dir}/")


if __name__ == "__main__":
    import numpy as np
    main()
