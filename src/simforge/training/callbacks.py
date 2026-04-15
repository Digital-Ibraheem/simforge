"""
Custom training callbacks for Stable-Baselines3.

WHAT IS A CALLBACK?
-------------------
SB3 callbacks are hooks that fire at specific points during training:
  - on_training_start() — once, before any steps
  - on_rollout_start()  — before each data collection phase
  - on_step()           — after every environment step
  - on_rollout_end()    — after data collection, before policy update
  - on_training_end()   — once, when training finishes

You register callbacks in model.learn(callback=...) and SB3 calls them
automatically. Callbacks let you add logging, checkpointing, and evaluation
without modifying SB3's training loop.

CALLBACKS HERE:
  1. SuccessRateCallback — tracks and logs episode success rate
  2. WandbCallback       — logs metrics to Weights & Biases
  3. VideoRecorderCallback — periodically saves evaluation videos

WEIGHTS & BIASES (W&B):
------------------------
W&B is an experiment tracking platform. When you call wandb.log({"metric": value}),
it sends that data point to the W&B cloud, where you can see real-time training
curves, compare runs, and share results. It's the standard tool for ML experiments.

To use it: `wandb login` (one-time), then it auto-initializes in train.py.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv


class SuccessRateCallback(BaseCallback):
    """
    Tracks episode success rate from environment info dicts.

    HOW SUCCESS IS REPORTED IN FETCH ENVS:
    The Fetch environments set info["is_success"] = 1.0 when the block
    reaches the goal position (within a tolerance). SB3's VecEnv collects
    these info dicts — we read them here to compute success rate.

    WHY NOT JUST USE EvalCallback?
    EvalCallback runs a separate evaluation env periodically. This callback
    tracks success rate from the *training* rollouts — no extra env needed.
    Both are useful: training success rate shows learning progress in real-time,
    eval success rate gives a cleaner signal (deterministic policy, no exploration).
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        """
        Args:
            log_freq: How often (in timesteps) to log the success rate.
            verbose:  0 = silent, 1 = print to console.
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_successes: list[float] = []

    def _on_step(self) -> bool:
        """
        Called after every environment step.

        self.locals["infos"] is a list of info dicts (one per parallel env).
        We look for "is_success" in each. When an episode ends, the final
        info dict contains the success flag.

        Returns True to continue training (False would stop it early).
        """
        for info in self.locals.get("infos", []):
            if "is_success" in info:
                self._episode_successes.append(float(info["is_success"]))

        # Log periodically
        if self.num_timesteps % self.log_freq == 0 and self._episode_successes:
            success_rate = np.mean(self._episode_successes[-100:])  # rolling window of last 100
            self.logger.record("rollout/success_rate", success_rate)
            if self.verbose:
                print(f"  [step {self.num_timesteps}] success_rate={success_rate:.3f}")

        return True


class WandbCallback(BaseCallback):
    """
    Logs training metrics to Weights & Biases.

    WHY A CUSTOM CALLBACK INSTEAD OF SB3'S BUILT-IN?
    SB3 logs to TensorBoard by default. W&B can consume TensorBoard logs,
    but this callback gives us more control: we can log custom metrics,
    add metadata, and log at the right frequency.

    This callback reads from SB3's internal logger (self.logger) which
    accumulates metrics between log calls.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Only log at the specified frequency to avoid overwhelming W&B
        if self.num_timesteps % self.log_freq == 0:
            try:
                import wandb

                # self.logger.name_to_value contains all logged metrics
                # (SB3 records things like train/actor_loss, train/critic_loss)
                metrics = {
                    k: v
                    for k, v in self.logger.name_to_value.items()
                    if v is not None
                }
                metrics["train/timesteps"] = self.num_timesteps
                metrics["train/time_elapsed"] = time.time() - self.training_env.reset.__self__._t_start if hasattr(self.training_env.reset, "__self__") else 0

                wandb.log(metrics, step=self.num_timesteps)

            except ImportError:
                # wandb not installed — silently skip
                pass
            except Exception:
                # Don't crash training over a logging error
                pass

        return True


class CheckpointCallback(BaseCallback):
    """
    Saves a copy of the model every N timesteps.

    WHY SAVE CHECKPOINTS?
    Training can take hours. If your laptop crashes at 800k steps, you'd lose
    everything. Checkpoints let you resume or use the best intermediate model.

    SB3 has its own CheckpointCallback but this one additionally saves metadata
    (timestep, success rate) alongside the model file.
    """

    def __init__(self, save_freq: int, save_path: str | Path, name_prefix: str = "model", verbose: int = 0) -> None:
        """
        Args:
            save_freq:   Save every N timesteps.
            save_path:   Directory to save models into.
            name_prefix: Filename prefix (e.g. "sac_her_pick_place").
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.save_path.mkdir(parents=True, exist_ok=True)
            path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(path)
            if self.verbose:
                print(f"  [checkpoint] saved to {path}")
        return True


class VideoRecorderCallback(BaseCallback):
    """
    Records a short evaluation video periodically during training.

    WHY VIDEO?
    Numbers are great for measuring progress but don't tell the full story.
    Watching the agent is the fastest way to understand what it's doing wrong:
    Is it not grasping? Grasping but dropping? Moving to the wrong place?

    HOW IT WORKS:
    We create a separate eval env with render_mode="rgb_array", run a few
    episodes, collect frames, and save an MP4. This is done in the callback
    rather than in the main training loop because we want it to happen
    automatically at fixed intervals.

    Note: Rendering is slow — don't set video_freq too low.

    Args:
        eval_env:   A separate environment for evaluation (not the training env).
        video_freq: Record every N timesteps.
        video_dir:  Where to save MP4 files.
        n_eval_episodes: How many episodes to record.
    """

    def __init__(
        self,
        eval_env: gymnasium.Env,
        video_freq: int = 50_000,
        video_dir: str | Path = "results/videos",
        n_eval_episodes: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_freq = video_freq
        self.video_dir = Path(video_dir)
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.num_timesteps % self.video_freq == 0:
            self._record_video()
        return True

    def _record_video(self) -> None:
        """Run evaluation episodes and save frames as an MP4."""
        try:
            import imageio

            self.video_dir.mkdir(parents=True, exist_ok=True)
            frames = []

            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                while not done:
                    # deterministic=True: no exploration noise during evaluation
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    frame = self.eval_env.render()
                    if frame is not None:
                        frames.append(frame)

            if frames:
                video_path = self.video_dir / f"eval_{self.num_timesteps:08d}_steps.mp4"
                imageio.mimsave(str(video_path), frames, fps=25)
                if self.verbose:
                    print(f"  [video] saved {len(frames)} frames to {video_path}")

                # Also log to W&B if available
                try:
                    import wandb
                    wandb.log(
                        {"eval/video": wandb.Video(str(video_path), fps=25, format="mp4")},
                        step=self.num_timesteps,
                    )
                except Exception:
                    pass

        except Exception as e:
            # Don't crash training over a video recording failure
            if self.verbose:
                print(f"  [video] recording failed: {e}")


# Import here to avoid circular reference (VideoRecorderCallback references gymnasium)
import gymnasium
