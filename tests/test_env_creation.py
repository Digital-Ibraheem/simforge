"""
Smoke tests: verify the environment and wrappers can be created.

These are fast sanity-check tests — they just verify nothing crashes at import
or creation time. Run these first before the full DR tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_fetch_env_registers():
    """gymnasium-robotics must register FetchPickAndPlace before gym.make() works."""
    try:
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401 — side effect: registers envs
        env = gym.make("FetchPickAndPlace-v4")
        obs, _ = env.reset()
        env.close()

        # Verify the observation has the expected structure
        assert "observation" in obs
        assert "achieved_goal" in obs
        assert "desired_goal" in obs
        assert obs["observation"].shape == (25,), f"Expected shape (25,), got {obs['observation'].shape}"
        assert obs["achieved_goal"].shape == (3,)
        assert obs["desired_goal"].shape == (3,)

    except ImportError as e:
        pytest.skip(f"gymnasium-robotics not installed: {e}")


def test_fetch_reach_env():
    """FetchReach should have 10-dim obs and 3-dim goals."""
    try:
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401
        env = gym.make("FetchReach-v4")
        obs, _ = env.reset()
        env.close()

        assert obs["observation"].shape == (10,)
        assert obs["achieved_goal"].shape == (3,)

    except ImportError as e:
        pytest.skip(f"gymnasium-robotics not installed: {e}")


def test_dense_reward_mode():
    """FetchPickAndPlace with reward_type='dense' should return negative-distance reward."""
    try:
        import gymnasium as gym
        import gymnasium_robotics  # noqa: F401
        env = gym.make("FetchPickAndPlace-v4", reward_type="dense")
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        env.close()

        # Dense reward = -distance, should be negative (block is not at goal)
        assert reward < 0, f"Expected negative dense reward, got {reward}"

    except ImportError as e:
        pytest.skip(f"gymnasium-robotics not installed: {e}")


def test_dr_wrapper_imports():
    """Our wrapper module imports should work without errors."""
    from simforge.envs.wrappers import DomainRandomizationWrapper, ObservationNoiseWrapper, ActionDelayWrapper
    from simforge.dr.params import DRParam, DistributionType
    from simforge.dr.randomizer import DomainRandomizer
    from simforge.dr.config import load_dr_config, build_params_from_config
