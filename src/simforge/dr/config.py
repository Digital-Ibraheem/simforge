"""
YAML configuration loading for domain randomization.

WHY YAML CONFIG FILES?
-----------------------
We could hardcode DR params in Python, but then comparing "no DR", "moderate DR",
and "aggressive DR" would require editing code. With YAML files, we can have:

    configs/dr_off.yaml       ← no randomization
    configs/dr_moderate.yaml  ← mild ranges
    configs/dr_aggressive.yaml ← wide ranges

And run three training jobs with different --config flags — no code changes needed.
This is standard practice in ML experiments for reproducibility.

SCHEMA:
-------
Each entry under 'domain_randomization' describes one DRParam.

Example YAML:
    domain_randomization:
      object_mass:
        body_name: "object0"
        mjfield: "body_mass"
        distribution: uniform
        low: 0.8
        high: 1.2
        is_multiplier: true
        requires_setconst: true

      table_friction:
        geom_name: "table0"
        mjfield: "geom_friction"
        component: 0          # 0=sliding, 1=torsional, 2=rolling
        distribution: uniform
        low: 0.6
        high: 1.4
        is_multiplier: true

      observation_noise:
        wrapper_type: obs_noise   # not a model param — handled by ObsNoiseWrapper
        distribution: gaussian
        low: 0.0                  # mean
        high: 0.005               # std

      action_delay:
        wrapper_type: action_delay
        delay_steps: 1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from simforge.dr.params import DRParam, DistributionType


def load_dr_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load and parse a domain randomization YAML file.

    Returns the raw parsed dict. Use build_params_from_config() to convert
    it to DRParam objects.

    Args:
        config_path: Path to the YAML file.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the config is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"DR config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Validate top-level structure
    if config is None or "domain_randomization" not in config:
        raise ValueError(
            f"Config at '{path}' must have a 'domain_randomization' top-level key. "
            f"Got: {list(config.keys()) if config else 'empty file'}"
        )

    return config


def build_params_from_config(config: dict[str, Any]) -> list[DRParam]:
    """
    Convert a parsed config dict into a list of DRParam objects.

    Each key under 'domain_randomization' becomes one DRParam.
    The key name is used as DRParam.name.

    Args:
        config: Dict produced by load_dr_config().

    Returns:
        List of DRParam instances ready to pass to DomainRandomizer.
    """
    dr_section = config["domain_randomization"]

    # If the section is empty or null (dr_off.yaml), return nothing
    if not dr_section:
        return []

    params = []
    for name, spec in dr_section.items():
        if spec is None:
            continue  # allow empty entries in YAML

        params.append(_build_param(name, spec))

    return params


def _build_param(name: str, spec: dict[str, Any]) -> DRParam:
    """
    Build a single DRParam from a YAML spec dict.

    The key mapping: YAML key → DRParam field. All fields have sensible
    defaults so minimal YAML is needed for simple cases.
    """
    # Parse distribution string → enum
    dist_str = spec.get("distribution", "uniform").lower()
    try:
        distribution = DistributionType(dist_str)
    except ValueError:
        valid = [d.value for d in DistributionType]
        raise ValueError(
            f"Parameter '{name}': unknown distribution '{dist_str}'. "
            f"Must be one of {valid}."
        )

    return DRParam(
        name=name,
        mjfield=spec.get("mjfield", ""),

        # Named accessors (look up index by name at runtime)
        body_name=spec.get("body_name"),
        geom_name=spec.get("geom_name"),
        actuator_name=spec.get("actuator_name"),

        # Raw index fallback
        index=spec.get("index"),

        # Which component of a multi-valued field (e.g. friction component 0=sliding)
        component=spec.get("component"),

        distribution=distribution,
        low=float(spec.get("low", 0.8)),
        high=float(spec.get("high", 1.2)),
        is_multiplier=bool(spec.get("is_multiplier", True)),

        # True for mass/inertia — must call mj_setConst after writing
        requires_setconst=bool(spec.get("requires_setconst", False)),

        # Non-model params: "obs_noise" or "action_delay"
        wrapper_type=spec.get("wrapper_type"),
    )


def get_wrapper_param(params: list[DRParam], wrapper_type: str) -> DRParam | None:
    """
    Find the DRParam for a specific wrapper type (e.g. "obs_noise").

    Used by ObservationNoiseWrapper and ActionDelayWrapper to read their
    configuration (std, delay_steps) from the same YAML config.

    Returns None if that wrapper type isn't configured (feature disabled).
    """
    for p in params:
        if p.wrapper_type == wrapper_type:
            return p
    return None
