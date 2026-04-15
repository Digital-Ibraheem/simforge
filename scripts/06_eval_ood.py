"""
Phase 3 (continued): Standalone OOD evaluation script.

PURPOSE:
--------
Run out-of-distribution evaluation on any saved model(s).
This is the script you run AFTER training to generate the ablation results.

You can use it to:
  - Check a single trained model's OOD robustness
  - Compare all ablation variants
  - Re-run evaluation with more episodes for tighter statistics

USAGE:
  # Evaluate one model on all OOD configs:
  python scripts/06_eval_ood.py --model results/models/ablation/ablation_no_dr_s42/best/best_model

  # Evaluate all ablation models and generate plots:
  python scripts/06_eval_ood.py --all-ablation

  # More episodes for tighter statistics:
  python scripts/06_eval_ood.py --all-ablation --n-episodes 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="OOD evaluation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--all-ablation", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all_ablation:
        results = _eval_all_ablation(args.n_episodes)
        _save_and_plot(results)
    elif args.model:
        _eval_one_model(args.model, args.n_episodes)
    else:
        print("Specify --model PATH or --all-ablation")
        sys.exit(1)


def _eval_one_model(model_path: str, n_episodes: int):
    """Evaluate one model across all OOD configs and print a table."""
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import yaml
    from stable_baselines3 import SAC
    from simforge.evaluation.evaluate import evaluate_under_ood

    with open(CONFIGS_DIR / "eval_ood.yaml") as f:
        ood_cfg = yaml.safe_load(f)

    env = gym.make("FetchPickAndPlace-v4")
    model = SAC.load(model_path, env=env)

    print(f"\nModel: {model_path}")
    print(f"{'Config':<22} {'Success Rate':>13} {'Mean Reward':>13}")
    print("-" * 50)

    for ood_config in ood_cfg["eval_configs"]:
        res = evaluate_under_ood(model, env, ood_config, n_episodes=n_episodes)
        print(f"  {ood_config['name']:<20} {res['success_rate']:>12.1%} {res['mean_reward']:>13.2f}")

    env.close()


def _eval_all_ablation(n_episodes: int) -> dict:
    """
    Evaluate all 3 ablation variants × 3 seeds on all OOD configs.

    Aggregates results across seeds (mean ± std).
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import yaml
    from simforge.evaluation.evaluate import evaluate_under_ood, load_model

    with open(CONFIGS_DIR / "eval_ood.yaml") as f:
        ood_cfg = yaml.safe_load(f)
    ood_configs = ood_cfg["eval_configs"]

    variants = {
        "no_dr": "ablation_no_dr",
        "moderate_dr": "ablation_moderate_dr",
        "aggressive_dr": "ablation_aggressive_dr",
    }
    seeds = [42, 1, 2]

    # results[variant][ood_config_name] = list of success_rates across seeds
    results_raw: dict[str, dict[str, list]] = {v: {c["name"]: [] for c in ood_configs} for v in variants}

    for variant, prefix in variants.items():
        for seed in seeds:
            model_path = RESULTS_DIR / "models" / "ablation" / f"{prefix}_s{seed}" / "best" / "best_model"
            if not model_path.with_suffix(".zip").exists():
                print(f"  Skipping {model_path} (not found)")
                continue

            print(f"\n  [{variant}, seed={seed}]")
            env = gym.make("FetchPickAndPlace-v4")
            model = load_model(model_path, env, algorithm="SAC")

            for ood_config in ood_configs:
                res = evaluate_under_ood(model, env, ood_config, n_episodes=n_episodes)
                results_raw[variant][ood_config["name"]].append(res["success_rate"])
                print(f"    {ood_config['name']:<20}: {res['success_rate']:.1%}")

            env.close()

    # Collapse to mean per variant per config
    results_mean: dict[str, dict[str, float]] = {}
    for variant in variants:
        results_mean[variant] = {}
        for cfg_name, rates in results_raw[variant].items():
            results_mean[variant][cfg_name] = float(np.mean(rates)) if rates else 0.0

    return results_mean


def _save_and_plot(results: dict):
    """Save results to JSON and generate plots."""
    out_dir = RESULTS_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw numbers (useful for the README table)
    json_path = out_dir / "ablation_ood_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # Print summary table
    print("\n--- Summary Table ---")
    variants = list(results.keys())
    configs = list(next(iter(results.values())).keys())

    header = f"{'Config':<22}" + "".join(f"{v:>16}" for v in variants)
    print(header)
    print("-" * len(header))
    for cfg in configs:
        row = f"{cfg:<22}" + "".join(f"{results[v].get(cfg, 0):.1%}".rjust(16) for v in variants)
        print(row)

    from simforge.evaluation.plotting import plot_ood_bar_chart, plot_heatmap
    plot_ood_bar_chart(results, out_dir / "ablation_ood_bar")
    plot_heatmap(results, out_dir / "ablation_heatmap")


if __name__ == "__main__":
    main()
