"""
Phase 4: Algorithm comparison — SAC+HER (sparse reward) vs PPO (dense reward).

PURPOSE:
--------
This experiment demonstrates WHY HER matters for sparse-reward manipulation.
We compare two fundamentally different approaches to the same task:

SAC + HER (off-policy, sparse reward):
  - The "right" approach for goal-conditioned manipulation.
  - Sparse reward: -1 every step, 0 when done. Hard to learn without HER.
  - HER relabels failed episodes as successes toward different goals.
  - Off-policy: learns from a replay buffer of past experience (sample-efficient).

PPO (on-policy, dense reward):
  - A general-purpose RL algorithm. No HER equivalent.
  - Must use dense reward: -distance_to_goal each step (Fetch envs support this natively).
  - On-policy: only learns from fresh data collected by the current policy.
  - More parallel envs (n_envs=8) to compensate for sample inefficiency.

EXPECTED RESULT:
  SAC+HER significantly outperforms PPO in:
    - Sample efficiency (reaches 50% success rate in far fewer timesteps)
    - Final performance (higher success rate at 1M steps)
  This is the standard result in the manipulation literature and makes
  a compelling story for the README.

WHY IS PPO WORSE HERE?
  1. Dense reward is a much weaker learning signal than HER for goal-conditioned tasks.
     The agent has to discover grasping, lifting, and placing from distance alone.
  2. PPO is on-policy: it must collect fresh data after every update, discarding
     all previous experience. This wastes data from early (random) episodes.
  3. Even with n_envs=8, PPO's sample efficiency is 5-10x worse than SAC on Fetch.

HOW TO COMPARE FAIRLY:
  - Same total timesteps (1M)
  - Same environment (FetchPickAndPlace-v4)
  - Same eval protocol (deterministic, 100 episodes)
  - Multiple seeds to account for variance

USAGE:
  python scripts/04_compare_algos.py                # train both algorithms
  python scripts/04_compare_algos.py --algo sac     # train only SAC+HER
  python scripts/04_compare_algos.py --algo ppo     # train only PPO
  python scripts/04_compare_algos.py --eval-only    # compare already-trained models
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simforge.training.train import TrainingConfig, train

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
RESULTS_DIR = Path(__file__).parent.parent / "results"

SEEDS = [42, 1, 2]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SAC+HER vs PPO")
    parser.add_argument("--algo", choices=["sac", "ppo", "both"], default="both")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.eval_only:
        if args.algo in ("sac", "both"):
            for seed in SEEDS:
                _train_sac(seed, args.timesteps, not args.no_wandb)

        if args.algo in ("ppo", "both"):
            for seed in SEEDS:
                _train_ppo(seed, args.timesteps, not args.no_wandb)

    print("\nGenerating comparison plots...")
    _generate_plots()


def _train_sac(seed: int, timesteps: int, use_wandb: bool):
    """Train SAC+HER — off-policy, sparse reward, HER relabeling."""
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
        dr_config=None,
        name=f"comparison_sac_her_s{seed}",
        model_dir=str(RESULTS_DIR / "models" / "comparison"),
        log_dir=str(RESULTS_DIR / "logs" / "comparison"),
        video_dir=str(RESULTS_DIR / "videos" / "comparison"),
        use_wandb=use_wandb,
        wandb_project="simforge",
        wandb_group="phase4-comparison",
        wandb_tags=["comparison", "sac-her"],
    )
    train(cfg)


def _train_ppo(seed: int, timesteps: int, use_wandb: bool):
    """
    Train PPO — on-policy, dense reward, parallel envs.

    Note reward_type="dense": Fetch envs natively support this.
    PPO cannot use HER (which requires a replay buffer, an off-policy concept).
    Dense reward is the standard substitute for sparse+HER when using PPO.
    """
    cfg = TrainingConfig(
        env_id="FetchPickAndPlace-v4",
        reward_type="dense",           # KEY DIFFERENCE: -distance_to_goal each step
        algorithm="PPO",
        use_her=False,                 # PPO is on-policy, HER doesn't apply
        n_envs=8,                      # parallel envs for faster data collection
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99,                    # higher discount for dense reward (longer horizon)
        n_steps=2048,                  # steps per env before each policy update
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        total_timesteps=timesteps,
        eval_freq=10_000,
        n_eval_episodes=20,
        seed=seed,
        dr_config=None,
        name=f"comparison_ppo_s{seed}",
        model_dir=str(RESULTS_DIR / "models" / "comparison"),
        log_dir=str(RESULTS_DIR / "logs" / "comparison"),
        video_dir=str(RESULTS_DIR / "videos" / "comparison"),
        use_wandb=use_wandb,
        wandb_project="simforge",
        wandb_group="phase4-comparison",
        wandb_tags=["comparison", "ppo"],
    )
    train(cfg)


def _generate_plots():
    """Load training logs and generate algorithm comparison plots."""
    import numpy as np
    from simforge.evaluation.plotting import plot_ood_bar_chart

    # Try to load eval results from saved model checkpoints
    results = {}

    for algo, name_prefix in [("sac_her", "comparison_sac_her"), ("ppo", "comparison_ppo")]:
        final_success_rates = []
        for seed in SEEDS:
            # Look for the eval log written by EvalCallback
            eval_log = RESULTS_DIR / "logs" / "comparison" / f"{name_prefix}_s{seed}" / "evaluations.npz"
            if eval_log.exists():
                data = np.load(eval_log)
                # data["results"] shape: (n_evals, n_eval_episodes)
                # data["timesteps"] shape: (n_evals,)
                final_sr = float(np.mean(data["successes"][-1]))
                final_success_rates.append(final_sr)

        if final_success_rates:
            results[algo] = {"FetchPickAndPlace\n(final 1M steps)": float(np.mean(final_success_rates))}

    if results:
        plot_dir = RESULTS_DIR / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_ood_bar_chart(
            results,
            save_path=plot_dir / "algo_comparison_final_success",
            title="Algorithm Comparison: Final Success Rate at 1M Steps",
        )


if __name__ == "__main__":
    main()
