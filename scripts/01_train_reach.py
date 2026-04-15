"""
Phase 1a: Train SAC+HER on FetchReach-v4.

PURPOSE:
--------
This is your warmup experiment. FetchReach is the simplest Fetch task —
just move the gripper to a floating target position. No grasping, no block.

Run this FIRST to:
  1. Verify the entire stack works: MuJoCo, gymnasium-robotics, SB3, W&B.
  2. See SAC+HER converge in ~15 minutes (not 3-6 hours).
  3. Understand what success looks like in the console and W&B dashboard.
  4. Get familiar with the observation/action space before pick-and-place.

WHAT TO WATCH:
  - `rollout/success_rate` should climb quickly from 0% to >90%
  - `train/actor_loss` and `train/critic_loss` should decrease over time
  - If success_rate stays near 0% after 20k steps, something is broken

USAGE:
  python scripts/01_train_reach.py
  python scripts/01_train_reach.py --no-wandb   # disable W&B logging
  python scripts/01_train_reach.py --seed 0     # different seed
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simforge.training.train import TrainingConfig, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC+HER on FetchReach-v4")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = TrainingConfig(
        # Task
        env_id="FetchReach-v4",
        reward_type="sparse",        # -1 each step, 0 at goal; HER handles this

        # Algorithm
        algorithm="SAC",
        use_her=True,

        # Hyperparameters — standard SB3 SAC+HER defaults for Fetch
        learning_rate=1e-3,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=1_000,
        gamma=0.95,
        tau=0.05,
        n_sampled_goal=4,
        goal_selection_strategy="future",

        # Training
        total_timesteps=args.timesteps,
        eval_freq=5_000,
        n_eval_episodes=20,
        seed=args.seed,

        # No domain randomization for the warmup
        dr_config=None,

        # Saving
        name=f"sac_her_reach_s{args.seed}",
        model_dir="results/models/reach",
        log_dir="results/logs/reach",
        video_dir="results/videos/reach",

        # W&B
        use_wandb=not args.no_wandb,
        wandb_project="simforge",
        wandb_group="phase1-reach",
        wandb_tags=["reach", "sac-her", "warmup"],
    )

    model = train(cfg)

    print("\n--- Quick evaluation ---")
    _run_quick_eval(model)


def _run_quick_eval(model):
    """
    Run 20 evaluation episodes and print the success rate.

    This gives you an immediate sanity check after training.
    The EvalCallback during training already does this periodically,
    but it's nice to see a final clean number printed at the end.
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import numpy as np

    env = gym.make("FetchReach-v4")
    successes = []

    for ep in range(20):
        obs, _ = env.reset()
        done = False
        while not done:
            # deterministic=True: use the mean of the policy distribution,
            # no exploration noise. Always use this for evaluation.
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        successes.append(info.get("is_success", False))

    success_rate = np.mean(successes)
    print(f"Final success rate over 20 eval episodes: {success_rate:.1%}")
    if success_rate < 0.8:
        print("WARNING: Success rate below 80%. Try training longer or check hyperparams.")
    else:
        print("Ready to move on to Phase 1b (FetchPickAndPlace).")


if __name__ == "__main__":
    main()
