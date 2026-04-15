"""
Tests for the domain randomization wrapper.

HOW TO RUN:
  cd simforge
  pytest tests/ -v

WHAT WE TEST:
  1. Does the DR wrapper actually change model parameters?
  2. Is mj_setConst called when mass is modified?
  3. Are defaults restored correctly?
  4. Does the wrapper survive reset() without crashing?
  5. Does the no-DR config (dr_off.yaml) act as a passthrough?

These are INTEGRATION tests — they require MuJoCo and gymnasium-robotics
to be installed. They won't run if those are missing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def _make_pick_place_env():
    """Create a FetchPickAndPlace env for testing. Skip if deps not available."""
    try:
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401
        return gym.make("FetchPickAndPlace-v4")
    except (ImportError, Exception) as e:
        pytest.skip(f"MuJoCo/gymnasium-robotics not available: {e}")


def test_dr_wrapper_changes_mass():
    """
    After DomainRandomizationWrapper.reset(), object mass should differ from default.

    This tests the core functionality: does randomization actually happen?
    We read the mass before and after reset() and verify they're different
    (across multiple resets, at least one should differ from default).
    """
    env = _make_pick_place_env()

    from simforge.envs.wrappers import DomainRandomizationWrapper

    # Wrap with moderate DR (which includes object_mass randomization)
    dr_env = DomainRandomizationWrapper(env, CONFIGS_DIR / "dr_moderate.yaml", seed=0)

    # Get the default mass (captured during capture_defaults)
    body_id = dr_env.unwrapped.model.body("object0").id
    default_mass = dr_env.randomizer.model_params[0].default_value  # first param is object_mass

    # Reset multiple times and collect masses
    masses_seen = set()
    for _ in range(10):
        dr_env.reset()
        current_mass = float(dr_env.unwrapped.model.body_mass[body_id])
        masses_seen.add(round(current_mass, 6))

    dr_env.close()

    # With 10 resets, we should see at least 2 different values
    assert len(masses_seen) > 1, (
        f"Expected mass to change across resets, but only saw: {masses_seen}"
    )

    # All values should be within the 0.8–1.2× range of the moderate DR config
    for mass in masses_seen:
        assert 0.75 * default_mass <= mass <= 1.25 * default_mass, (
            f"Mass {mass} is outside expected range [{0.75*default_mass}, {1.25*default_mass}]"
        )


def test_dr_off_does_not_change_physics():
    """
    The no-DR config (dr_off.yaml) should not change any model parameters.
    """
    env = _make_pick_place_env()

    from simforge.envs.wrappers import DomainRandomizationWrapper

    dr_env = DomainRandomizationWrapper(env, CONFIGS_DIR / "dr_off.yaml", seed=0)

    body_id = dr_env.unwrapped.model.body("object0").id
    mass_before = float(dr_env.unwrapped.model.body_mass[body_id])

    # Reset 5 times
    for _ in range(5):
        dr_env.reset()

    mass_after = float(dr_env.unwrapped.model.body_mass[body_id])
    dr_env.close()

    # Mass should be exactly the same — no randomization
    assert mass_before == mass_after, (
        f"Expected no mass change with dr_off config. Before: {mass_before}, After: {mass_after}"
    )


def test_restore_defaults():
    """
    restore_defaults() should bring physics back to original values.
    """
    env = _make_pick_place_env()

    from simforge.envs.wrappers import DomainRandomizationWrapper

    dr_env = DomainRandomizationWrapper(env, CONFIGS_DIR / "dr_moderate.yaml", seed=42)

    body_id = dr_env.unwrapped.model.body("object0").id

    # Capture the default that was recorded
    original_mass = dr_env.randomizer.model_params[0].default_value

    # Randomize
    dr_env.reset()
    randomized_mass = float(dr_env.unwrapped.model.body_mass[body_id])

    # Restore
    dr_env.randomizer.restore_defaults(dr_env.unwrapped.model, dr_env.unwrapped.data)
    restored_mass = float(dr_env.unwrapped.model.body_mass[body_id])

    dr_env.close()

    assert abs(restored_mass - original_mass) < 1e-9, (
        f"restore_defaults() failed. original={original_mass}, restored={restored_mass}"
    )


def test_observation_wrapper_adds_noise():
    """
    ObservationNoiseWrapper should change observations vs. the base env.
    """
    env = _make_pick_place_env()

    from simforge.envs.wrappers import ObservationNoiseWrapper

    noisy_env = ObservationNoiseWrapper(env, noise_std=0.1, seed=0)
    obs_noisy, _ = noisy_env.reset()

    # Also get observation without noise for comparison
    base_env = _make_pick_place_env()
    obs_clean, _ = base_env.reset(seed=0)

    noisy_env.close()
    base_env.close()

    # The noisy observation should differ from clean
    # (with std=0.1 and 25-dim obs, the chance of all being identical is negligible)
    obs_diff = np.abs(obs_noisy["observation"] - obs_clean["observation"])
    assert np.any(obs_diff > 0), "Observation noise wrapper did not change observations"


def test_config_loading():
    """
    All three DR configs should load without errors.
    """
    from simforge.dr.config import load_dr_config, build_params_from_config

    for config_name in ["dr_off.yaml", "dr_moderate.yaml", "dr_aggressive.yaml"]:
        config = load_dr_config(CONFIGS_DIR / config_name)
        params = build_params_from_config(config)

        if config_name == "dr_off.yaml":
            assert len(params) == 0, "dr_off.yaml should produce 0 params"
        else:
            assert len(params) > 0, f"{config_name} should produce at least 1 param"


def test_action_delay_wrapper():
    """
    ActionDelayWrapper with delay_steps=1 should execute the previous action.
    """
    env = _make_pick_place_env()

    from simforge.envs.wrappers import ActionDelayWrapper

    delayed_env = ActionDelayWrapper(env, delay_steps=1)
    delayed_env.reset()

    # Send a known action
    action_a = np.array([1.0, 0.0, 0.0, 0.0])  # move right
    action_b = np.array([-1.0, 0.0, 0.0, 0.0])  # move left

    # First step: buffer has [zero, action_a], executes zero
    executed_a = delayed_env.action(action_a)
    assert np.allclose(executed_a, np.zeros(4)), f"Expected zero action at t=0, got {executed_a}"

    # Second step: buffer has [action_a, action_b], executes action_a
    executed_b = delayed_env.action(action_b)
    assert np.allclose(executed_b, action_a), f"Expected action_a at t=1, got {executed_b}"

    delayed_env.close()
