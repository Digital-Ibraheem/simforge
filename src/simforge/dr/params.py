"""
Domain randomization parameter definitions.

This module defines the data structures that describe a single randomizable
parameter — what MuJoCo field to modify, what distribution to sample from,
and what the original (default) value was.

Design principle: this module has ZERO imports from the rest of simforge.
That makes it easy to extract into a standalone library in Phase 6 without
any dependency untangling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DistributionType(Enum):
    """
    How to sample a randomized value.

    UNIFORM:     sample uniformly between [low, high].
                 Good for bounded physical parameters (friction, mass multiplier).

    GAUSSIAN:    sample from a normal distribution N(mean, std).
                 Good when you want a "most likely" center value with tails.
                 Use low=mean, high=std in the config.

    LOGUNIFORM:  sample uniformly in log-space between [low, high].
                 Good when the parameter spans orders of magnitude (e.g. stiffness
                 might reasonably range from 1 to 1000).
    """
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    LOGUNIFORM = "loguniform"


@dataclass
class DRParam:
    """
    A single domain-randomizable parameter.

    Each instance describes one thing we want to randomize (e.g. "object mass"),
    where to find it in the MuJoCo model, how to sample it, and what its
    original value was.

    Fields:
        name:              Human-readable label, used in logs and plots.
        mjfield:           The mjModel array to write to, e.g. "body_mass".
                           For nested fields like opt.gravity, use "opt.gravity".
        body_name:         If set, we look up the body index by name at runtime.
                           Preferred over raw indices — survives model changes.
        geom_name:         Same but for geoms.
        actuator_name:     Same but for actuators.
        index:             Raw integer index, used if no name is provided.
        component:         Which element of a multi-valued field to modify.
                           e.g. geom_friction[i] has 3 components: [sliding, torsional, rolling].
                           Set component=0 for sliding friction.
        distribution:      Which distribution to sample from (see DistributionType).
        low:               Lower bound (uniform) or mean (gaussian).
        high:              Upper bound (uniform) or std (gaussian).
        is_multiplier:     If True, the sampled value is multiplied by the default
                           instead of used directly. e.g. low=0.8, high=1.2 means
                           "80%–120% of the original mass". More robust than absolute
                           values because it works regardless of the original model's units.
        requires_setconst: If True, mujoco.mj_setConst(model, data) must be called
                           after modifying this param. Required for mass and inertia
                           changes — MuJoCo caches derived constants that go stale.
        default_value:     The original value from the model, captured at wrapper init.
                           Populated automatically by DomainRandomizer.capture_defaults().
        wrapper_type:      For non-model params (observation noise, action delay),
                           set this to "obs_noise" or "action_delay". These are handled
                           by separate wrapper classes, not by writing to mjModel.
    """
    name: str
    mjfield: str = ""

    # Named accessors (preferred over raw indices)
    body_name: Optional[str] = None
    geom_name: Optional[str] = None
    actuator_name: Optional[str] = None

    # Fallback: raw integer index into the mjModel array
    index: Optional[int] = None

    # Which component of a multi-valued field (e.g. friction has 3 components)
    component: Optional[int] = None

    distribution: DistributionType = DistributionType.UNIFORM
    low: float = 0.8
    high: float = 1.2
    is_multiplier: bool = True       # treat low/high as ratios of default?

    requires_setconst: bool = False  # must call mj_setConst after this change?

    # Populated at runtime by DomainRandomizer.capture_defaults()
    default_value: float = field(default=0.0, init=False)

    # If set, this param is handled by a wrapper, not by writing to mjModel
    wrapper_type: Optional[str] = None  # "obs_noise" | "action_delay" | None


def sample_value(param: DRParam, rng: "np.random.Generator") -> float:  # type: ignore[name-defined]
    """
    Draw one sample from the parameter's configured distribution.

    Kept as a standalone function (not a method) so it's easy to test and reuse.
    """
    import numpy as np

    if param.distribution == DistributionType.UNIFORM:
        raw = rng.uniform(param.low, param.high)
    elif param.distribution == DistributionType.GAUSSIAN:
        # low=mean, high=std for gaussian
        raw = rng.normal(param.low, param.high)
    elif param.distribution == DistributionType.LOGUNIFORM:
        # Sample uniformly in log space, then exponentiate
        raw = np.exp(rng.uniform(np.log(param.low), np.log(param.high)))
    else:
        raise ValueError(f"Unknown distribution: {param.distribution}")

    if param.is_multiplier:
        # Scale relative to the original value captured at init
        return param.default_value * raw
    return raw
