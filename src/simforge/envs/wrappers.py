"""
Gymnasium wrappers for domain randomization.

WHAT IS A GYMNASIUM WRAPPER?
-----------------------------
A wrapper is a class that wraps an existing environment and modifies its
behavior without changing the environment itself. It intercepts calls to
reset(), step(), and other methods, adds its own logic, then delegates to
the real environment.

Think of it like middleware: env → Wrapper1 → Wrapper2 → your training loop.

All wrappers here inherit from gymnasium.Wrapper (or sub-types):
  - gymnasium.Wrapper:            can override reset(), step(), etc.
  - gymnasium.ObservationWrapper: intercepts observations (override observation())
  - gymnasium.ActionWrapper:      intercepts actions (override action())
  - gymnasium.RewardWrapper:      intercepts rewards (override reward())

WHY WRAPPERS INSTEAD OF SUBCLASSING THE ENV DIRECTLY?
Composability. You can stack wrappers:
    env = gym.make("FetchPickAndPlace-v4")
    env = DomainRandomizationWrapper(env, config)
    env = ObservationNoiseWrapper(env, std=0.01)
    env = ActionDelayWrapper(env, delay_steps=1)

Each wrapper does one thing. This is much cleaner than one giant class
that does everything.

WRAPPERS IN THIS FILE:
  1. DomainRandomizationWrapper — randomizes MuJoCo model physics each episode
  2. ObservationNoiseWrapper    — adds Gaussian noise to observations
  3. ActionDelayWrapper         — delays actions by N steps
  4. DenseRewardWrapper         — note: not needed for Fetch envs (use reward_type="dense")
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium
import numpy as np

from simforge.dr.config import load_dr_config, build_params_from_config, get_wrapper_param
from simforge.dr.randomizer import DomainRandomizer


# =============================================================================
# 1. Domain Randomization Wrapper
# =============================================================================

class DomainRandomizationWrapper(gymnasium.Wrapper):
    """
    Randomizes MuJoCo physics parameters at the start of each episode.

    HOW IT WORKS:
    On every call to reset(), this wrapper:
      1. Calls the real env's reset() first (which resets the simulator state).
      2. Then writes randomized physics values into the MuJoCo model.
      3. Returns the observation from the now-randomized environment.

    WHY RANDOMIZE AFTER reset()?
    gymnasium-robotics Fetch environments call mujoco.mj_resetData() during
    their reset(), which can restore certain model fields. If we randomized
    BEFORE reset(), our changes would get overwritten. Randomizing after
    reset() ensures our changes persist through the episode.

    The policy trains across thousands of these randomly-varied episodes and
    learns to succeed despite the variation — giving it robustness.
    """

    def __init__(self, env: gymnasium.Env, config_path: str | Path, seed: int | None = None) -> None:
        """
        Args:
            env:         The environment to wrap (e.g. from gym.make()).
            config_path: Path to a DR YAML config file.
            seed:        Optional seed for the randomization RNG.
                         Each episode draws new samples from this seeded generator,
                         giving reproducible randomization sequences.
        """
        super().__init__(env)

        # Load config and build the randomizer
        config = load_dr_config(config_path)
        self.randomizer = DomainRandomizer.from_config(config)

        # numpy's new Generator API — thread-safe, faster, better statistics
        # than the legacy np.random.uniform
        self.rng = np.random.default_rng(seed)

        # Capture original model values now (before any randomization)
        # env.unwrapped gives us the base env, bypassing any intermediate wrappers
        self.randomizer.capture_defaults(self.env.unwrapped.model)

        # Also store wrapper params for obs noise / action delay in case someone
        # reads them (used by ObservationNoiseWrapper/ActionDelayWrapper)
        all_params = build_params_from_config(config)
        self._obs_noise_param = get_wrapper_param(all_params, "obs_noise")
        self._action_delay_param = get_wrapper_param(all_params, "action_delay")

    def reset(self, **kwargs) -> tuple:
        """
        Reset the environment, then immediately randomize physics.

        The **kwargs are passed through to the base env's reset() so that
        seed, options, etc. work normally.
        """
        # Step 1: Let the real environment reset (clears physics state, re-places objects)
        obs, info = self.env.reset(**kwargs)

        # Step 2: Overwrite physics params with fresh random samples
        # model = the static physics description (masses, frictions, gains...)
        # data  = the dynamic simulation state (positions, velocities...)
        self.randomizer.randomize(
            self.env.unwrapped.model,
            self.env.unwrapped.data,
            self.rng,
        )

        return obs, info


# =============================================================================
# 2. Observation Noise Wrapper
# =============================================================================

class ObservationNoiseWrapper(gymnasium.ObservationWrapper):
    """
    Adds Gaussian noise to the robot's state observations.

    WHY? In the real world, sensors have noise. A robot's joint encoders
    don't report exact positions — they have small errors. If the policy
    trains on perfect observations, it may be sensitive to sensor noise
    when deployed on real hardware.

    By adding noise during training, we teach the policy to act robustly
    even when observations are slightly wrong.

    WHAT GETS NOISE ADDED?
    Only the "observation" key of the dict obs — this is the robot's state
    (gripper position, velocities, block pose, etc.).
    We do NOT add noise to "achieved_goal" or "desired_goal" — these are
    logical goal specifications, not noisy sensor readings.

    OBSERVATION SPACE FOR FETCH ENVS:
    The Fetch environments return a dict with 3 keys:
      - "observation":    Box(25,) — robot state + object state
          [0:3]   grip_pos: gripper position in world frame
          [3:6]   object_pos: block position in world frame
          [6:9]   object_rel_pos: block position relative to gripper
          [9:12]  gripper_state: finger positions (2 values, but padded to 3)
          [12:15] object_rot: block rotation (Euler angles)
          [15:18] object_velp: block linear velocity
          [18:21] object_velr: block angular velocity
          [21:24] grip_velp: gripper linear velocity
          [24]    gripper_vel: finger velocity
      - "achieved_goal":  Box(3,) — current block position (goal achieved so far)
      - "desired_goal":   Box(3,) — target block position

    Args:
        env:   The environment to wrap.
        noise_std: Standard deviation of added Gaussian noise.
                   0.005 (5mm) is a reasonable starting point.
        seed:  Optional seed for the noise RNG.
    """

    def __init__(self, env: gymnasium.Env, noise_std: float = 0.005, seed: int | None = None) -> None:
        super().__init__(env)
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def observation(self, obs: dict) -> dict:
        """
        Called automatically by gymnasium.ObservationWrapper after every step() and reset().
        We add noise to the observation vector and return the modified dict.
        """
        if self.noise_std > 0.0:
            # Add Gaussian noise to robot/object state only
            obs["observation"] = obs["observation"] + self.rng.normal(
                loc=0.0,
                scale=self.noise_std,
                size=obs["observation"].shape,
            )
        return obs


# =============================================================================
# 3. Action Delay Wrapper
# =============================================================================

class ActionDelayWrapper(gymnasium.ActionWrapper):
    """
    Delays actions by N timesteps to simulate control latency.

    WHY? Real robot control loops have latency: the policy computes an action,
    sends it over a network or serial bus, and the actuator executes it
    ~20–100ms later. During that delay, the robot keeps moving. If the
    policy wasn't trained with this delay, it may oscillate or overshoot
    when deployed on real hardware.

    HOW IT WORKS:
    We maintain a circular buffer (deque) of past actions with capacity
    delay_steps + 1. When the policy submits action a_t, we actually
    execute the action from delay_steps timesteps ago (a_{t-delay}).

    At the start of each episode, the buffer is filled with zero actions
    (no actuation) to represent the "warmup" period.

    Example with delay_steps=2:
        t=0: policy sends a0, buffer=[0, 0, a0], execute 0  (from 2 steps ago)
        t=1: policy sends a1, buffer=[0, a0, a1], execute 0
        t=2: policy sends a2, buffer=[a0, a1, a2], execute a0
        t=3: policy sends a3, buffer=[a1, a2, a3], execute a1

    Args:
        env:          The environment to wrap.
        delay_steps:  How many timesteps to delay. 0 = no delay (passthrough).
                      1–2 steps is realistic for a well-engineered system.
    """

    def __init__(self, env: gymnasium.Env, delay_steps: int = 1) -> None:
        super().__init__(env)
        self.delay_steps = max(0, delay_steps)

        # Zero action to fill the buffer during warmup
        if hasattr(env.action_space, "shape"):
            self._zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            self._zero_action = 0

        # deque with maxlen automatically discards the oldest entry when full
        self._buffer: deque = deque(
            [self._zero_action.copy() for _ in range(self.delay_steps + 1)],
            maxlen=self.delay_steps + 1,
        )

    def reset(self, **kwargs) -> tuple:
        """Refill the buffer with zero actions at episode start."""
        obs, info = self.env.reset(**kwargs)
        self._buffer = deque(
            [self._zero_action.copy() for _ in range(self.delay_steps + 1)],
            maxlen=self.delay_steps + 1,
        )
        return obs, info

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Called by gymnasium.ActionWrapper before every step().

        Pushes the new action into the buffer and returns the delayed action
        from delay_steps timesteps ago.

        The deque's maxlen automatically discards entries beyond the buffer size.
        """
        if self.delay_steps == 0:
            return action

        # Add the new action to the right of the deque
        self._buffer.append(action)
        # The oldest entry (left side) is the one to execute now
        delayed_action = self._buffer[0]
        return delayed_action


# =============================================================================
# Convenience factory
# =============================================================================

def build_dr_env(base_env: gymnasium.Env, config_path: str | Path, seed: int | None = None) -> gymnasium.Env:
    """
    Apply all DR-related wrappers from a single config file.

    This reads the config once and stacks the appropriate wrappers:
      1. DomainRandomizationWrapper (always, even if config has no model params)
      2. ObservationNoiseWrapper    (only if obs_noise is in the config)
      3. ActionDelayWrapper         (only if action_delay is in the config)

    Args:
        base_env:    The raw gymnasium environment (from gym.make()).
        config_path: Path to the DR YAML config.
        seed:        Optional base seed for all RNGs.

    Returns:
        The wrapped environment, ready for training.
    """
    config = load_dr_config(config_path)
    all_params = build_params_from_config(config)

    # Always apply the model-level randomizer (even if empty — no-ops safely)
    env = DomainRandomizationWrapper(base_env, config_path, seed=seed)

    # Add obs noise if configured
    obs_noise_param = get_wrapper_param(all_params, "obs_noise")
    if obs_noise_param is not None:
        noise_std = obs_noise_param.high  # high=std for gaussian
        env = ObservationNoiseWrapper(env, noise_std=noise_std, seed=(seed or 0) + 1)

    # Add action delay if configured
    action_delay_param = get_wrapper_param(all_params, "action_delay")
    if action_delay_param is not None:
        delay = int(action_delay_param.low)  # stored in low field for delay_steps
        env = ActionDelayWrapper(env, delay_steps=delay)

    return env
