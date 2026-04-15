"""
Domain randomization parameter definitions.

WHAT IS DOMAIN RANDOMIZATION?
-------------------------------
When you train a robot in simulation, the simulation has exact physics:
object mass = 0.1kg, friction = 0.5, gravity = -9.81 m/s². But the real
world is messier — the block might weigh 0.13kg today, 0.09kg tomorrow
depending on how it was manufactured; the table might be slightly slippery.

If you train on exact values, the policy "memorizes" those values and fails
when anything differs. Domain randomization (DR) is the fix: during training,
you randomly vary these physical parameters every episode. The policy has to
learn to succeed across a *range* of physics, not one specific setting. This
makes it more robust when deployed.

WHY THIS FILE?
--------------
This module defines *what* can be randomized — the data structure for a
single randomizable parameter. The actual randomization logic lives in
randomizer.py; this file is just the schema/types.

DESIGN PRINCIPLE: Zero imports from the rest of simforge.
This module (and randomizer.py and config.py) are intentionally self-contained.
In Phase 6, we'll extract them into a standalone PyPI library. If they imported
from other simforge modules, that extraction would require untangling dependencies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DistributionType(Enum):
    """
    The probability distribution used to sample a randomized value.

    WHY THREE DISTRIBUTIONS?

    UNIFORM: "anything between X and Y is equally likely."
        Good for most physical parameters where you have a reasonable range.
        Example: mass multiplier between 0.8x and 1.2x.

    GAUSSIAN: "values near the center are most likely, with a normal bell curve."
        Good when you expect a "true" value with random measurement error.
        Example: gravity ≈ -9.81 with std=0.3 (slight variation from local geology).
        When using GAUSSIAN, 'low' = mean, 'high' = std in the config.

    LOGUNIFORM: "sample uniformly in log-space."
        Good when a parameter spans orders of magnitude and all scales matter equally.
        Example: joint stiffness can range from 1 to 1000. Uniform sampling would
        spend 99.9% of samples above 1.0, rarely exploring the low end. Log-uniform
        spreads samples evenly across 1–10, 10–100, 100–1000.
    """
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    LOGUNIFORM = "loguniform"


@dataclass
class DRParam:
    """
    A single domain-randomizable parameter.

    Think of each DRParam as a recipe card for randomizing one thing:
    "For object mass: find the body named 'object0', read its current mass,
    and each episode multiply it by a random value between 0.8 and 1.2."

    HOW MUJOCO STORES PHYSICS PARAMETERS:
    MuJoCo stores the entire physics model in a flat C struct called mjModel.
    Every body, geom, joint, and actuator in the simulation has an integer ID,
    and its properties are stored in arrays indexed by that ID.

    Example: if "object0" has body ID 5, then:
        model.body_mass[5]        = 0.1   (mass in kg)
        model.body_inertia[5]     = [...]  (3x3 inertia tensor)
        model.body_pos[5]         = [x,y,z] (position in world frame)

    To randomize mass, we write a new value to model.body_mass[5].
    To know which index to use, we look up "object0" by name.

    Fields:
        name:
            Human-readable label for this parameter. Used in logs, plots,
            and error messages. Example: "object_mass", "table_friction".

        mjfield:
            The mjModel attribute to modify. Examples:
              - "body_mass"         → model.body_mass[id]
              - "geom_friction"     → model.geom_friction[id, component]
              - "opt.gravity"       → model.opt.gravity[component]
              - "actuator_gainprm"  → model.actuator_gainprm[id, component]
            For nested fields (like opt.gravity), we use "opt.gravity" and
            handle the dot-traversal in the randomizer.

        body_name / geom_name / actuator_name:
            The string name of the body/geom/actuator to modify.
            We look up the integer ID at runtime using these names.
            STRONGLY preferred over raw 'index' — if the MJCF model changes
            and body IDs shift, names still work correctly.

        index:
            Fallback: directly specify the integer array index.
            Only use this if the element has no name.

        component:
            Some mjModel fields are multi-valued per element.
            Example: geom_friction[i] has 3 values:
              [0] = sliding friction (most important, resists lateral movement)
              [1] = torsional friction (resists spinning about contact normal)
              [2] = rolling friction (resists rolling)
            Set component=0 to randomize only sliding friction.
            Set component=None to randomize all components with the same sample.

        distribution:
            Which distribution to sample from (see DistributionType above).

        low / high:
            The two parameters of the distribution:
              - UNIFORM:    low=minimum, high=maximum
              - GAUSSIAN:   low=mean, high=std
              - LOGUNIFORM: low=log-space minimum, high=log-space maximum

        is_multiplier:
            The most important design choice in this schema.

            If True (default): low/high are *ratios* of the original value.
              Example: low=0.8, high=1.2 → "80% to 120% of the original mass"
              This is robust because it works regardless of the model's units
              or actual values. A 0.8–1.2x range makes physical sense whether
              the original mass is 0.01kg or 10kg.

            If False: low/high are *absolute* values in the model's units.
              Example: low=-10.0, high=-9.0 → gravity between -10 and -9 m/s².
              Use this for parameters where you know the exact range you want.

        requires_setconst:
            A subtle but CRITICAL MuJoCo internals detail.

            MuJoCo precomputes certain "derived constants" from the model
            (like composite inertia terms) and stores them in mjModel for
            performance. If you change body_mass or body_inertia at runtime,
            these derived constants go stale — the simulation will use the
            new mass for some calculations but the old cached inertia for others.

            The fix: after changing mass or inertia, call:
                mujoco.mj_setConst(model, data)
            This recomputes all derived constants from scratch.

            Set requires_setconst=True for any param that touches:
              - body_mass
              - body_inertia
            Leave it False for friction, gravity, actuator gains (no caching).

        default_value:
            The original value from the model, captured when the wrapper is
            first created. This is what is_multiplier=True multiplies against.
            You don't set this manually — DomainRandomizer.capture_defaults()
            reads it from the live model.

        wrapper_type:
            Some "domain randomization" isn't about MuJoCo model parameters
            at all. Observation noise and action delay are implemented as
            Python-level wrappers that intercept step() and reset() calls.
            Set wrapper_type="obs_noise" or "action_delay" for these.
            The DomainRandomizer will skip them (they're handled by other wrappers).
    """
    name: str
    mjfield: str = ""

    # Named accessors (preferred)
    body_name: Optional[str] = None
    geom_name: Optional[str] = None
    actuator_name: Optional[str] = None

    # Fallback: raw integer index
    index: Optional[int] = None

    # Which element of a multi-valued field (e.g. friction[0]=sliding)
    component: Optional[int] = None

    distribution: DistributionType = DistributionType.UNIFORM
    low: float = 0.8
    high: float = 1.2

    # If True, sample is used as a multiplier on the default value
    is_multiplier: bool = True

    # If True, must call mujoco.mj_setConst(model, data) after writing
    requires_setconst: bool = False

    # Captured at runtime by DomainRandomizer.capture_defaults()
    default_value: float = field(default=0.0, init=False)

    # "obs_noise" or "action_delay" → handled by wrapper classes, not mjModel writes
    wrapper_type: Optional[str] = None


def sample_value(param: DRParam, rng: "np.random.Generator") -> float:  # type: ignore[name-defined]
    """
    Draw one sample from the parameter's configured distribution.

    WHY A STANDALONE FUNCTION?
    Methods belong to a class and couple logic to data. This function only
    needs the param schema and an RNG — making it standalone means it's easy
    to unit-test in isolation and reuse in other contexts.

    Args:
        param: The DRParam describing what to sample.
        rng:   A numpy random Generator (e.g. np.random.default_rng(42)).
               We use the new Generator API (not np.random.uniform) because
               it's faster, statistically better, and avoids global state.

    Returns:
        A float ready to write to the mjModel array.
        If param.is_multiplier=True, this already includes the default_value scaling.
    """
    import numpy as np

    if param.distribution == DistributionType.UNIFORM:
        raw = rng.uniform(param.low, param.high)

    elif param.distribution == DistributionType.GAUSSIAN:
        # Convention: low=mean, high=std
        raw = rng.normal(param.low, param.high)

    elif param.distribution == DistributionType.LOGUNIFORM:
        # Transform: sample u ~ Uniform(log(low), log(high)), return exp(u)
        # This gives uniform coverage in log-space
        raw = float(np.exp(rng.uniform(np.log(param.low), np.log(param.high))))

    else:
        raise ValueError(f"Unknown distribution type: {param.distribution}")

    if param.is_multiplier:
        # Scale relative to the original value captured from the model
        return param.default_value * raw
    return raw
