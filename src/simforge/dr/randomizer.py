"""
The core domain randomization engine.

WHAT THIS MODULE DOES:
----------------------
Given a list of DRParam definitions, this class:
  1. At initialization: reads the original values from the live MuJoCo model
     (so is_multiplier=True has a reference to scale against).
  2. At each episode reset: samples new values for all params and writes them
     into the MuJoCo model arrays.
  3. Handles the mj_setConst requirement for mass/inertia changes.
  4. Can restore defaults (useful for evaluation on default physics).

MUJOCO MODEL vs DATA — CRITICAL DISTINCTION:
--------------------------------------------
MuJoCo separates simulation state into two structs:

  mjModel (model): The static description of the robot/world.
    - Body masses, geom sizes, joint limits, actuator gains...
    - Things that describe *what* the simulation is, not its current state.
    - Accessed via env.unwrapped.model

  mjData (data): The dynamic state at the current timestep.
    - Joint positions (qpos), velocities (qvel), applied forces...
    - Things that change every simulation step.
    - Accessed via env.unwrapped.data

Domain randomization modifies mjModel (the physics description).
The simulation loop reads from mjModel to know how physics work.

ZERO IMPORTS FROM SIMFORGE:
----------------------------
This file only imports from: mujoco, numpy, and params.py (same package).
This keeps the DR engine self-contained for Phase 6 extraction into a
standalone PyPI library.
"""

from __future__ import annotations

import mujoco
import numpy as np

from simforge.dr.params import DRParam, sample_value


class DomainRandomizer:
    """
    Applies domain randomization to a MuJoCo model at each episode reset.

    Usage:
        randomizer = DomainRandomizer(params)
        randomizer.capture_defaults(env.unwrapped.model)   # call once at init

        # Then every episode reset:
        randomizer.randomize(env.unwrapped.model, env.unwrapped.data, rng)
    """

    def __init__(self, params: list[DRParam]) -> None:
        """
        Args:
            params: List of DRParam instances describing what to randomize.
                    Usually loaded from a YAML config via load_dr_config().
        """
        # Filter out wrapper-level params (obs noise, action delay) — those are
        # handled by separate Gymnasium wrappers, not by writing to mjModel.
        self.model_params = [p for p in params if p.wrapper_type is None]

        # Check once whether any param needs mj_setConst so we don't check per-step
        self._needs_setconst = any(p.requires_setconst for p in self.model_params)

        # Also keep wrapper params accessible for the wrapper classes to read
        self.wrapper_params = {p.name: p for p in params if p.wrapper_type is not None}

        self._defaults_captured = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def capture_defaults(self, model) -> None:
        """
        Read the original parameter values from the live MuJoCo model.

        WHY: DRParam.default_value needs to be populated before we can apply
        is_multiplier=True randomization (we need the baseline to multiply).
        We can't do this in DRParam.__init__ because the model doesn't exist yet
        at config-load time — it only exists after gym.make() creates the env.

        Call this exactly once, right after creating the environment.
        """
        for param in self.model_params:
            idx = self._resolve_index(model, param)
            param.default_value = self._read_field(model, param.mjfield, idx, param.component)

        self._defaults_captured = True

    # ------------------------------------------------------------------
    # Per-episode randomization
    # ------------------------------------------------------------------

    def randomize(self, model, data, rng: np.random.Generator) -> None:
        """
        Sample new parameter values and write them into the MuJoCo model.

        Call this AFTER env.reset() completes — gymnasium-robotics resets
        the model state during reset(), which would overwrite our changes
        if we randomized before reset.

        Args:
            model: env.unwrapped.model  (the mjModel struct)
            data:  env.unwrapped.data   (the mjData struct, needed for mj_setConst)
            rng:   A numpy Generator for sampling
        """
        if not self._defaults_captured:
            raise RuntimeError(
                "capture_defaults() must be called before randomize(). "
                "Call it once after creating the environment."
            )

        for param in self.model_params:
            idx = self._resolve_index(model, param)
            new_value = sample_value(param, rng)
            self._write_field(model, param.mjfield, idx, param.component, new_value)

        # After writing mass/inertia, recompute MuJoCo's derived constants.
        # Skip if no param needs it — mj_setConst is relatively cheap but not free.
        if self._needs_setconst:
            mujoco.mj_setConst(model, data)

    def restore_defaults(self, model, data) -> None:
        """
        Reset all parameters back to their original (default) values.

        Useful when evaluating in-distribution performance: you want the model
        to behave exactly as it did before any randomization.
        """
        if not self._defaults_captured:
            return

        for param in self.model_params:
            idx = self._resolve_index(model, param)
            self._write_field(model, param.mjfield, idx, param.component, param.default_value)

        if self._needs_setconst:
            mujoco.mj_setConst(model, data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_index(self, model, param: DRParam) -> int | None:
        """
        Convert a named body/geom/actuator into its integer array index.

        MuJoCo uses named entities in MJCF XML files, but stores everything
        as integer-indexed flat arrays at runtime. This translates between them.
        """
        if param.body_name is not None:
            return model.body(param.body_name).id
        elif param.geom_name is not None:
            return model.geom(param.geom_name).id
        elif param.actuator_name is not None:
            return model.actuator(param.actuator_name).id
        elif param.index is not None:
            return param.index
        else:
            # For global fields like opt.gravity, there's no per-element index
            return None

    def _read_field(self, model, mjfield: str, idx: int | None, component: int | None) -> float:
        """
        Read the current value of a MuJoCo model field.

        Handles two types of fields:
          - "body_mass"    → model.body_mass[idx]
          - "opt.gravity"  → model.opt.gravity[component]
          - "geom_friction" → model.geom_friction[idx, component]
        """
        arr = self._get_array(model, mjfield)

        if idx is None and component is None:
            # Scalar field (unusual case)
            return float(arr)
        elif idx is None:
            # Global array indexed only by component (e.g. opt.gravity[2])
            return float(arr[component])
        elif component is None:
            # Per-entity scalar (e.g. body_mass[id])
            return float(arr[idx])
        else:
            # Per-entity vector (e.g. geom_friction[id, 0])
            return float(arr[idx, component])

    def _write_field(self, model, mjfield: str, idx: int | None, component: int | None, value: float) -> None:
        """Write a new value to the MuJoCo model field."""
        arr = self._get_array(model, mjfield)

        if idx is None and component is None:
            arr[()] = value
        elif idx is None:
            arr[component] = value
        elif component is None:
            arr[idx] = value
        else:
            arr[idx, component] = value

    def _get_array(self, model, mjfield: str):
        """
        Navigate from a dot-separated field string to the actual numpy array.

        Examples:
            "body_mass"    → model.body_mass       (direct attribute)
            "opt.gravity"  → model.opt.gravity     (nested: model.opt then .gravity)
            "geom_friction"→ model.geom_friction

        MuJoCo exposes its internal arrays as numpy views, so writing to them
        directly modifies the underlying C memory — which is exactly what we want.
        """
        parts = mjfield.split(".")
        obj = model
        for part in parts:
            obj = getattr(obj, part)
        return obj

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "DomainRandomizer":
        """
        Build a DomainRandomizer from a parsed YAML config dict.

        The config dict has the structure produced by load_dr_config().
        See config.py for the full schema.
        """
        from simforge.dr.config import build_params_from_config
        params = build_params_from_config(config)
        return cls(params)
