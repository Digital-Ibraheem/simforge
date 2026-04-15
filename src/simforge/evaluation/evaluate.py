"""
Evaluation framework for trained policies.

WHY A SEPARATE EVALUATION MODULE?
-----------------------------------
Training and evaluation have different goals:
  - Training: maximize reward over time, with exploration noise (stochastic policy)
  - Evaluation: measure true performance, no noise (deterministic policy)

This module handles evaluation: running a trained model against an environment
and collecting metrics. It's used by:
  - Phase 3 ablation: compare no-DR / moderate-DR / aggressive-DR policies
  - Phase 4 algo comparison: compare SAC+HER vs PPO policies
  - The EvalCallback during training (but that uses SB3's built-in evaluate_policy)

DETERMINISTIC vs STOCHASTIC POLICY:
  SAC learns a *stochastic* policy: an action distribution (Gaussian) over actions.
  During training, it samples from this distribution (exploration).
  During evaluation, we use the *mean* of the distribution (deterministic=True).
  This removes noise from evaluation, giving a cleaner performance signal.

METRICS WE TRACK:
  - success_rate:  fraction of episodes where info["is_success"]=True (main metric)
  - mean_reward:   average total reward per episode (secondary)
  - episode_length: average steps per episode (shorter = more efficient)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def evaluate_policy(
    model,
    env,
    n_episodes: int = 100,
    deterministic: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run n_episodes evaluation episodes and return aggregate metrics.

    This is a simple, transparent implementation you can read and trust.
    SB3 has its own evaluate_policy() which does the same thing, but this
    version explicitly returns success_rate (which SB3's doesn't by default).

    Args:
        model:         A trained SB3 model (SAC, PPO, etc.)
        env:           A Gymnasium environment (not VecEnv — single env for clarity)
        n_episodes:    How many episodes to run. 100 is enough for low-variance estimates.
        deterministic: True = use mean policy (no exploration). Always True for eval.
        verbose:       True = print per-episode results.

    Returns:
        Dict with keys:
            "success_rate":   float in [0, 1]
            "mean_reward":    float
            "std_reward":     float
            "mean_ep_length": float
            "all_successes":  list[bool]   — per-episode result (for statistics)
            "all_rewards":    list[float]  — per-episode total reward
    """
    successes = []
    rewards = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            # model.predict() returns (action, state)
            # state is only non-None for recurrent policies (LSTM etc.) — ignore it here
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        # "is_success" is set in the final info dict when the episode ends
        # It's 1.0 (float) when the goal was achieved, 0.0 otherwise
        success = bool(info.get("is_success", False))
        successes.append(success)
        rewards.append(episode_reward)
        lengths.append(episode_length)

        if verbose:
            status = "SUCCESS" if success else "fail"
            print(f"  ep {ep+1:3d}: {status:7s} | reward={episode_reward:.1f} | len={episode_length}")

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_ep_length": float(np.mean(lengths)),
        "all_successes": successes,
        "all_rewards": rewards,
    }


def evaluate_under_ood(
    model,
    env,
    ood_config: dict,
    n_episodes: int = 100,
) -> dict:
    """
    Evaluate a policy under a specific out-of-distribution physics config.

    Applies physics overrides directly to the MuJoCo model, runs evaluation,
    then restores defaults. This lets us measure robustness without training
    a separate policy for each OOD condition.

    OOD CONFIG FORMAT (from eval_ood.yaml):
        physics_overrides:
          object_mass_multiplier: 3.0    → multiply original mass by 3
          table_friction_multiplier: 0.3  → multiply original friction by 0.3
          gravity_z: -1.62               → set gravity to absolute value

    Args:
        model:      Trained policy.
        env:        A Gymnasium env (with access to env.unwrapped.model).
        ood_config: One entry from the eval_configs list in eval_ood.yaml.
        n_episodes: How many episodes to evaluate.

    Returns:
        Same dict as evaluate_policy() plus "ood_config_name".
    """
    import mujoco

    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    # --- Store defaults ---
    # We capture the current values so we can restore them afterward.
    # This is important because the same env object might be reused across
    # multiple OOD configs.
    overrides = ood_config.get("physics_overrides", {})
    saved = {}

    # Apply overrides
    if "object_mass_multiplier" in overrides:
        body_id = mj_model.body("object0").id
        saved["body_mass"] = float(mj_model.body_mass[body_id])
        mj_model.body_mass[body_id] = saved["body_mass"] * overrides["object_mass_multiplier"]
        mujoco.mj_setConst(mj_model, mj_data)

    if "table_friction_multiplier" in overrides:
        geom_id = mj_model.geom("table0").id
        saved["geom_friction"] = float(mj_model.geom_friction[geom_id, 0])
        mj_model.geom_friction[geom_id, 0] = saved["geom_friction"] * overrides["table_friction_multiplier"]

    if "gravity_z" in overrides:
        saved["gravity_z"] = float(mj_model.opt.gravity[2])
        mj_model.opt.gravity[2] = overrides["gravity_z"]

    # --- Run evaluation ---
    results = evaluate_policy(model, env, n_episodes=n_episodes)
    results["ood_config_name"] = ood_config.get("name", "unknown")

    # --- Restore defaults ---
    if "body_mass" in saved:
        body_id = mj_model.body("object0").id
        mj_model.body_mass[body_id] = saved["body_mass"]
        mujoco.mj_setConst(mj_model, mj_data)

    if "geom_friction" in saved:
        geom_id = mj_model.geom("table0").id
        mj_model.geom_friction[geom_id, 0] = saved["geom_friction"]

    if "gravity_z" in saved:
        mj_model.opt.gravity[2] = saved["gravity_z"]

    return results


def load_model(model_path: str | Path, env, algorithm: str = "SAC"):
    """
    Load a saved SB3 model from disk.

    SB3 saves models as zip files containing:
      - policy weights (PyTorch state dict)
      - model hyperparameters (so you don't need to specify them again)
      - optionally the replay buffer (large — not saved by default)

    Args:
        model_path: Path to the saved model (with or without .zip extension).
        env:        An environment to attach to the model (for action/obs space).
        algorithm:  "SAC" or "PPO".

    Returns:
        The loaded SB3 model, ready for model.predict().
    """
    from stable_baselines3 import SAC, PPO

    path = Path(model_path)
    cls = {"SAC": SAC, "PPO": PPO}[algorithm]
    return cls.load(path, env=env)
