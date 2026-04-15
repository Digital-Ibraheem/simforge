"""
Phase 1b: Train SAC+HER on FetchPickAndPlace-v4.

PURPOSE:
--------
This is the main experiment. Train the core pick-and-place agent WITHOUT
domain randomization. This gives you:
  1. A baseline policy to compare against DR variants in Phase 3.
  2. Hands-on experience watching a difficult RL task converge over hours.
  3. A trained model you can load and visualize.

TRAINING TIME:
  ~3-6 hours on CPU (Apple Silicon). The bottleneck is MuJoCo simulation,
  not neural network computation — SAC is mostly CPU-bound.
  If you have a GPU, the speedup is modest (~20-30%) for this task size.

WHAT TO WATCH IN THE W&B DASHBOARD:
  - rollout/success_rate: climbs slowly at first (~0% for 100-200k steps),
    then starts rising as the agent learns to grasp
  - train/actor_loss: should trend downward
  - train/critic_loss: may spike, then stabilize
  - Evaluation videos: watch the agent improve over time

WHY IT TAKES SO LONG:
  Pick-and-place requires learning a multi-step skill:
    1. Move gripper above block
    2. Lower to block height
    3. Close gripper (grasp)
    4. Lift block
    5. Move to target position
    6. Release
  Each step is only rewarded if ALL steps are done correctly (sparse reward).
  HER relabeling helps, but the agent still needs to discover grasping by
  exploring randomly, which is rare.

USAGE:
  python scripts/02_train_pick_place.py                  # standard run
  python scripts/02_train_pick_place.py --seed 1         # different seed
  python scripts/02_train_pick_place.py --timesteps 2000000  # longer run
  python scripts/02_train_pick_place.py --resume results/models/pick_place/sac_her_pp_s42/checkpoints/...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simforge.training.train import TrainingConfig, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC+HER on FetchPickAndPlace-v4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a saved model checkpoint to resume training from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = TrainingConfig(
        env_id="FetchPickAndPlace-v4",
        reward_type="sparse",

        algorithm="SAC",
        use_her=True,

        # These are the same hyperparameters as Reach — SAC+HER is robust
        # to hyperparameter choices on Fetch tasks. These are well-validated defaults.
        learning_rate=1e-3,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=1_000,
        gamma=0.95,
        tau=0.05,
        n_sampled_goal=4,
        goal_selection_strategy="future",

        total_timesteps=args.timesteps,
        eval_freq=10_000,
        n_eval_episodes=20,
        seed=args.seed,

        dr_config=None,            # no DR for the baseline

        name=f"sac_her_pp_no_dr_s{args.seed}",
        model_dir="results/models/pick_place",
        log_dir="results/logs/pick_place",
        video_dir="results/videos/pick_place",

        use_wandb=not args.no_wandb,
        wandb_project="simforge",
        wandb_group="phase1-pick-place",
        wandb_tags=["pick-and-place", "sac-her", "no-dr"],
    )

    if args.resume:
        # Load a saved model and continue training from it.
        # Useful if you need to stop and resume a long training run.
        from stable_baselines3 import SAC
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401

        print(f"Resuming from checkpoint: {args.resume}")
        env = gym.make(cfg.env_id, reward_type=cfg.reward_type)
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        vec_env = VecMonitor(DummyVecEnv([lambda: env]))
        model = SAC.load(args.resume, env=vec_env)
        # Note: when resuming, the replay buffer is not saved by default.
        # The model will retrain from scratch on that front, but the policy
        # weights carry over. For full resume, use model.save() which can
        # optionally save the buffer: model.save(..., include=["replay_buffer"])
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,  # continue step counter from checkpoint
        )
    else:
        train(cfg)


if __name__ == "__main__":
    main()
