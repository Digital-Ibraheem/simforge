"""
Unified training entrypoint.

WHY A SHARED TRAIN FUNCTION?
-----------------------------
All our training scripts (reach, pick-and-place, ablation, PPO comparison)
share the same structure:
  1. Create and wrap the environment
  2. Instantiate the model with hyperparameters
  3. Register callbacks
  4. Call model.learn()
  5. Save the model

Rather than repeating this in every script, we centralize it here. Individual
scripts (in scripts/) just build a config and call train().

STABLE-BASELINES3 OVERVIEW:
-----------------------------
SB3 is a library of clean, reliable RL algorithm implementations. We use:

  SAC (Soft Actor-Critic):
    - An off-policy algorithm — it learns from stored past experience (replay buffer).
    - "Off-policy" means it can reuse old data, making it sample-efficient.
    - Adds entropy maximization: the policy is trained not just to maximize reward
      but also to stay random/exploratory. This helps it avoid local optima.
    - Best for continuous action spaces (like robot control).

  HER (Hindsight Experience Replay):
    - A technique that works ON TOP of SAC (not instead of it).
    - The pick-and-place task has SPARSE reward: -1 at every step, 0 when done.
      With sparse reward, the agent almost never succeeds by random exploration,
      so it gets almost no positive signal to learn from.
    - HER's insight: even failed episodes are useful. After an episode where the
      block ended up at position [0.5, 0.3, 0.4] (not the goal), we can relabel
      that episode as if [0.5, 0.3, 0.4] *was* the goal. Now the agent "succeeded"!
    - By relabeling, HER creates a stream of artificial successes that teach the
      policy to control the block, even before it ever reaches the real goal.
    - HerReplayBuffer is in stable_baselines3 (not sb3-contrib) as of SB3 v2.

  PPO (Proximal Policy Optimization):
    - An on-policy algorithm — it only learns from fresh experience collected by
      the current policy.
    - "On-policy" means it can't reuse old data, so it's less sample-efficient.
    - But: generally more stable and easier to tune than off-policy methods.
    - Can't use HER (HER requires an off-policy replay buffer).
    - Needs DENSE reward to work on pick-and-place (we use reward_type="dense").

HYPERPARAMETER GUIDE (SAC+HER):
    learning_rate: How fast to update the neural network weights. 1e-3 is standard.
    batch_size: How many transitions to sample from the replay buffer per update.
                Larger = more stable gradients, but slower per step.
    buffer_size: Max capacity of the replay buffer (number of transitions).
                 1M is standard for manipulation tasks.
    learning_starts: Don't update the policy until this many steps are collected.
                     Lets the buffer fill up with diverse data before training.
    gamma: Discount factor. 0.95 means "reward in 20 steps is worth 0.95^20 ≈ 36%
           of reward now". Lower gamma = shorter planning horizon. 0.95 is good
           for short-horizon tasks like Fetch (max 50 steps per episode).
    tau: How quickly the target network tracks the online network. 0.05 = slow
         tracking. Target networks stabilize training in off-policy methods.
    n_sampled_goal: How many HER relabeled goals to generate per real transition.
                    4 is standard. More = stronger HER signal, more computation.
    goal_selection_strategy: "future" = relabel with goals achieved LATER in the
                             same episode. Best empirical performance for Fetch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers Fetch envs
import numpy as np

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from simforge.training.callbacks import (
    SuccessRateCallback,
    WandbCallback,
    CheckpointCallback,
    VideoRecorderCallback,
)
from simforge.utils.helpers import make_env


@dataclass
class TrainingConfig:
    """
    All configuration for a single training run.

    Using a dataclass keeps configs explicit and type-checkable.
    Scripts construct one of these and pass it to train().

    Fields are grouped by concern:
      - env:        which environment and how to create it
      - algorithm:  which RL algorithm
      - hp:         hyperparameters
      - her:        HER-specific settings (only used with SAC)
      - logging:    where to save and how often to log
      - training:   total timesteps, eval frequency, seeds
    """
    # Environment
    env_id: str = "FetchPickAndPlace-v4"
    reward_type: str = "sparse"         # "sparse" (for SAC+HER) or "dense" (for PPO)
    n_envs: int = 1                     # parallel envs (PPO benefits from >1)
    dr_config: Optional[str] = None     # path to DR YAML, or None for no DR

    # Algorithm: "SAC" or "PPO"
    algorithm: str = "SAC"

    # Shared hyperparameters
    learning_rate: float = 1e-3
    gamma: float = 0.95                 # discount factor
    batch_size: int = 256
    seed: int = 42

    # SAC-specific
    buffer_size: int = 1_000_000
    learning_starts: int = 1_000
    tau: float = 0.05
    train_freq: int = 1
    gradient_steps: int = 1

    # HER-specific (only used when algorithm="SAC")
    use_her: bool = True
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"

    # PPO-specific
    n_steps: int = 2048
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01

    # Training duration
    total_timesteps: int = 1_000_000

    # Evaluation
    eval_freq: int = 10_000            # evaluate every N timesteps
    n_eval_episodes: int = 20          # episodes per evaluation

    # Saving and logging
    model_dir: str = "results/models"
    log_dir: str = "results/logs"
    video_dir: str = "results/videos"
    name: str = "run"                  # used in filenames and W&B run name

    # W&B
    use_wandb: bool = True
    wandb_project: str = "simforge"
    wandb_group: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)


def train(cfg: TrainingConfig) -> Any:
    """
    Run a full training job with the given config.

    Steps:
      1. Initialize W&B (if enabled)
      2. Create training environment (with DR if configured)
      3. Create eval environment (always uses default physics for fair comparison)
      4. Instantiate the model (SAC+HER or PPO)
      5. Register callbacks
      6. Train
      7. Save final model

    Args:
        cfg: TrainingConfig describing what to train.

    Returns:
        The trained SB3 model object.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {cfg.name}")
    print(f"  Algorithm: {cfg.algorithm} | Env: {cfg.env_id}")
    print(f"  Total timesteps: {cfg.total_timesteps:,}")
    print(f"  DR config: {cfg.dr_config or 'none'}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Initialize W&B
    # ------------------------------------------------------------------
    if cfg.use_wandb:
        _init_wandb(cfg)

    # ------------------------------------------------------------------
    # 2. Create training environment
    # ------------------------------------------------------------------
    # VecEnv = "Vectorized Environment" — SB3 requires this wrapper.
    # Even with n_envs=1, we need DummyVecEnv (runs envs sequentially in one process).
    # SubprocVecEnv runs envs in separate processes (faster for PPO with n_envs>1).

    def _make_train_env():
        """Factory for one training env instance."""
        env = gym.make(cfg.env_id, reward_type=cfg.reward_type)
        if cfg.dr_config:
            from simforge.envs.wrappers import build_dr_env
            env = build_dr_env(env, cfg.dr_config, seed=cfg.seed)
        return env

    if cfg.n_envs == 1:
        train_env = DummyVecEnv([_make_train_env])
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        train_env = SubprocVecEnv([
            make_env(cfg.env_id, seed=cfg.seed, rank=i, reward_type=cfg.reward_type)
            for i in range(cfg.n_envs)
        ])

    # VecMonitor wraps VecEnv to automatically record episode stats
    # (episode length, reward) that SB3's logger reads
    train_env = VecMonitor(train_env)

    # ------------------------------------------------------------------
    # 3. Create evaluation environment
    # ------------------------------------------------------------------
    # IMPORTANT: the eval env uses DEFAULT physics (no DR) so we measure
    # performance on the standard task, regardless of training conditions.
    # This makes ablation comparisons fair.

    eval_env = DummyVecEnv([make_env(cfg.env_id, seed=cfg.seed + 1000, reward_type=cfg.reward_type)])
    eval_env = VecMonitor(eval_env)

    # Separate single env for video recording (needs render_mode="rgb_array")
    video_env = gym.make(cfg.env_id, render_mode="rgb_array")

    # ------------------------------------------------------------------
    # 4. Instantiate the model
    # ------------------------------------------------------------------
    model = _make_model(cfg, train_env)

    # ------------------------------------------------------------------
    # 5. Register callbacks
    # ------------------------------------------------------------------
    model_path = Path(cfg.model_dir) / cfg.name
    log_path = Path(cfg.log_dir) / cfg.name
    model_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Track success rate from training rollouts
        SuccessRateCallback(log_freq=1000, verbose=1),

        # Evaluate on clean env periodically and save best model
        # EvalCallback is SB3's built-in: runs n_eval_episodes, logs mean_reward
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_path / "best"),
            log_path=str(log_path),
            eval_freq=cfg.eval_freq,
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,  # no exploration noise during eval
            verbose=1,
        ),

        # Save checkpoints so we can resume if training crashes
        CheckpointCallback(
            save_freq=50_000,
            save_path=model_path / "checkpoints",
            name_prefix=cfg.name,
            verbose=1,
        ),

        # Record video every 50k steps
        VideoRecorderCallback(
            eval_env=video_env,
            video_freq=50_000,
            video_dir=Path(cfg.video_dir) / cfg.name,
            n_eval_episodes=1,
            verbose=1,
        ),
    ]

    # Add W&B metric logging
    if cfg.use_wandb:
        callbacks.append(WandbCallback(log_freq=1000, verbose=0))

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    t_start = time.time()
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=CallbackList(callbacks),
        log_interval=10,         # print SB3's built-in console log every 10 episodes
        reset_num_timesteps=True,
        progress_bar=False,
    )
    elapsed = time.time() - t_start
    print(f"\nTraining done in {elapsed/3600:.1f}h ({elapsed:.0f}s)")

    # ------------------------------------------------------------------
    # 7. Save final model
    # ------------------------------------------------------------------
    final_path = model_path / "final_model"
    model.save(final_path)
    print(f"Final model saved to: {final_path}")

    # Finish W&B run
    if cfg.use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    return model


def _make_model(cfg: TrainingConfig, train_env):
    """
    Instantiate the right SB3 model class from the config.

    WHY "MultiInputPolicy"?
    The Fetch environments return DICT observations (not a flat array).
    SB3's "MultiInputPolicy" is designed for dict obs — it runs each key
    through a small MLP encoder and concatenates the results before the
    main policy network. You'd use "MlpPolicy" for flat array obs.
    """
    if cfg.algorithm == "SAC":
        kwargs = dict(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            gamma=cfg.gamma,
            tau=cfg.tau,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            verbose=1,
            seed=cfg.seed,
            tensorboard_log=f"results/logs/tb_{cfg.name}",
        )

        if cfg.use_her:
            # HER wraps the replay buffer — SAC stores transitions through HER,
            # and HER automatically adds relabeled versions when sampling batches.
            from stable_baselines3 import HerReplayBuffer
            kwargs["replay_buffer_class"] = HerReplayBuffer
            kwargs["replay_buffer_kwargs"] = dict(
                n_sampled_goal=cfg.n_sampled_goal,
                goal_selection_strategy=cfg.goal_selection_strategy,
            )

        return SAC(**kwargs)

    elif cfg.algorithm == "PPO":
        return PPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            verbose=1,
            seed=cfg.seed,
            tensorboard_log=f"results/logs/tb_{cfg.name}",
        )

    else:
        raise ValueError(f"Unknown algorithm '{cfg.algorithm}'. Must be 'SAC' or 'PPO'.")


def _init_wandb(cfg: TrainingConfig) -> None:
    """
    Initialize a Weights & Biases run.

    Each training run gets its own W&B run, named cfg.name. The group
    lets you compare multiple runs (e.g. all seeds of the same variant)
    side-by-side in the W&B UI.
    """
    try:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.name,
            group=cfg.wandb_group,
            tags=cfg.wandb_tags,
            config={
                # Log all hyperparameters for reproducibility
                "env_id": cfg.env_id,
                "algorithm": cfg.algorithm,
                "total_timesteps": cfg.total_timesteps,
                "learning_rate": cfg.learning_rate,
                "gamma": cfg.gamma,
                "batch_size": cfg.batch_size,
                "seed": cfg.seed,
                "dr_config": cfg.dr_config,
                "use_her": cfg.use_her,
                "n_sampled_goal": cfg.n_sampled_goal if cfg.use_her else None,
            },
        )
    except ImportError:
        print("wandb not installed. Skipping W&B logging.")
        cfg.use_wandb = False
    except Exception as e:
        print(f"W&B init failed: {e}. Skipping.")
        cfg.use_wandb = False
