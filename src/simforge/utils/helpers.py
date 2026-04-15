"""
General-purpose utilities used across training scripts.

These are small helpers that don't belong in any single module — things like
creating environments with consistent seeding, or resolving body/geom names
to their integer indices in the MuJoCo model.
"""

import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — importing this registers the Fetch envs with gymnasium
import numpy as np


def make_env(env_id: str, seed: int | None = None, rank: int = 0, **kwargs):
    """
    Return a factory function that creates a single Gymnasium environment.

    Why a factory? SB3's VecEnv helpers (make_vec_env) expect a callable
    that produces an env, not the env itself. This pattern also makes it
    easy to create multiple envs with different seeds (seed + rank).

    Args:
        env_id: Gymnasium env ID, e.g. "FetchPickAndPlace-v4"
        seed: Base random seed. Each parallel env gets seed + rank.
        rank: Index offset so parallel envs get different seeds.
        **kwargs: Passed through to gym.make() (e.g. reward_type="dense").
    """
    def _init() -> gym.Env:
        env = gym.make(env_id, **kwargs)
        if seed is not None:
            # Seeding gives reproducible episodes — important for comparing runs
            env.reset(seed=seed + rank)
        return env

    return _init


def find_body_id(model, body_name: str) -> int:
    """
    Return the integer index of a named body in the MuJoCo model.

    MuJoCo stores everything in flat arrays indexed by integers. When we want
    to change the mass of "object0", we need to know which row of
    model.body_mass[] that corresponds to. This function does that lookup.

    Raises ValueError if the name doesn't exist (prevents silent wrong-index bugs).
    """
    # model.body() returns a named accessor; .id gives the integer index
    try:
        return model.body(body_name).id
    except KeyError:
        raise ValueError(
            f"Body '{body_name}' not found in model. "
            f"Available bodies: {[model.body(i).name for i in range(model.nbody)]}"
        )


def find_geom_id(model, geom_name: str) -> int:
    """Return the integer index of a named geom in the MuJoCo model."""
    try:
        return model.geom(geom_name).id
    except KeyError:
        raise ValueError(
            f"Geom '{geom_name}' not found in model. "
            f"Available geoms: {[model.geom(i).name for i in range(model.ngeom)]}"
        )


def find_actuator_id(model, actuator_name: str) -> int:
    """Return the integer index of a named actuator in the MuJoCo model."""
    try:
        return model.actuator(actuator_name).id
    except KeyError:
        raise ValueError(
            f"Actuator '{actuator_name}' not found in model. "
            f"Available actuators: {[model.actuator(i).name for i in range(model.nu)]}"
        )


def set_random_seed(seed: int) -> None:
    """
    Seed numpy's global RNG.

    SB3 handles its own internal seeding via env.reset(seed=...) and model
    constructor arguments. This call is for any numpy operations in our own
    code (e.g. sampling OOD eval configs).
    """
    np.random.seed(seed)
