"""
Phase 5: Record demo videos of trained policies.

PURPOSE:
--------
Videos are essential for communicating what the agent actually does.
Reading success_rate=0.78 tells you the agent is decent; watching it
grasp and place a block makes it real and presentable.

Record videos for:
  1. A successful pick-and-place episode (the main demo)
  2. A few interesting failure modes (shows honest analysis)
  3. Side-by-side: no-DR policy failing on OOD vs DR policy succeeding

WHAT YOU GET:
  results/videos/demos/
    best_policy_success.mp4   — clean success on standard physics
    failure_low_friction.mp4  — no-DR policy failing on slippery surface
    dr_success_low_friction.mp4 — DR policy succeeding on same condition
    ...

USAGE:
  # Record demo for a specific model:
  python scripts/05_record_demo.py --model results/models/pick_place/sac_her_pp_no_dr_s42/best/best_model

  # Record demos for all ablation models:
  python scripts/05_record_demo.py --all-ablation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

RESULTS_DIR = Path(__file__).parent.parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Record evaluation videos")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model")
    parser.add_argument("--all-ablation", action="store_true", help="Record all ablation models")
    parser.add_argument("--n-episodes", type=int, default=5, help="Episodes to record")
    parser.add_argument("--fps", type=int, default=25, help="Video frame rate")
    parser.add_argument("--width", type=int, default=640, help="Video width")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument(
        "--ood",
        type=str,
        default=None,
        choices=["heavy_object", "low_friction", "moon_gravity", "combined_hard"],
        help="Apply an OOD physics config while recording",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all_ablation:
        _record_all_ablation(args)
    elif args.model:
        _record_single(args.model, args)
    else:
        print("Specify --model PATH or --all-ablation")
        sys.exit(1)


def _record_single(model_path: str, args, tag: str = "demo") -> Path:
    """
    Record n_episodes from a single model and save as MP4.

    Returns:
        Path to the saved video file.
    """
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401
    import imageio
    import mujoco
    import numpy as np
    from stable_baselines3 import SAC

    video_dir = RESULTS_DIR / "videos" / "demos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create the env with rgb_array rendering
    # render_mode="rgb_array" means env.render() returns a numpy array instead of opening a window
    env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array",
                   width=args.width, height=args.height)

    # Apply OOD physics if requested
    if args.ood:
        env = _apply_ood_physics(env, args.ood)

    model = SAC.load(model_path, env=env)

    all_frames = []
    successes = 0

    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        frames = []
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Render a frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        success = bool(info.get("is_success", False))
        if success:
            successes += 1

        status = "SUCCESS" if success else "fail"
        print(f"  Episode {ep+1}: {status} | reward={episode_reward:.1f} | frames={len(frames)}")
        all_frames.extend(frames)

    env.close()

    # Save video
    ood_suffix = f"_{args.ood}" if args.ood else ""
    video_name = Path(model_path).parent.parent.name + ood_suffix
    video_path = video_dir / f"{video_name}_{tag}.mp4"

    if all_frames:
        imageio.mimsave(str(video_path), all_frames, fps=args.fps)
        print(f"  Saved {len(all_frames)} frames → {video_path}")
        print(f"  Success rate: {successes}/{args.n_episodes}")
    else:
        print("  WARNING: No frames captured. Check render_mode.")

    return video_path


def _apply_ood_physics(env, ood_name: str):
    """
    Apply hard-coded OOD physics to the environment.

    This is a simplified version of evaluate_under_ood() for recording purposes.
    We modify the model once at the start and leave it changed for the whole recording
    (since we want to demonstrate behavior under consistent OOD conditions).
    """
    import mujoco

    # We need to get model access — wrap to ensure we can do it after reset
    class OODWrapper(type(env).__bases__[0] if hasattr(type(env), '__bases__') else object):
        pass

    # Directly modify after first reset
    original_reset = env.reset

    def patched_reset(**kwargs):
        result = original_reset(**kwargs)
        model = env.unwrapped.model
        data = env.unwrapped.data

        if ood_name == "heavy_object":
            body_id = model.body("object0").id
            model.body_mass[body_id] *= 3.0
            mujoco.mj_setConst(model, data)
        elif ood_name == "low_friction":
            geom_id = model.geom("table0").id
            model.geom_friction[geom_id, 0] *= 0.3
        elif ood_name == "moon_gravity":
            model.opt.gravity[2] = -1.62
        elif ood_name == "combined_hard":
            body_id = model.body("object0").id
            model.body_mass[body_id] *= 2.0
            mujoco.mj_setConst(model, data)
            geom_id = model.geom("table0").id
            model.geom_friction[geom_id, 0] *= 0.5
            model.opt.gravity[2] = -7.0

        return result

    env.reset = patched_reset
    return env


def _record_all_ablation(args):
    """Record videos for all 3 ablation variants (using seed 42 model for each)."""
    variants = ["no_dr", "moderate_dr", "aggressive_dr"]

    for variant in variants:
        model_path = RESULTS_DIR / "models" / "ablation" / f"ablation_{variant}_s42" / "best" / "best_model"
        if not model_path.with_suffix(".zip").exists():
            print(f"Model not found: {model_path} — skipping")
            continue

        print(f"\n--- Recording {variant} ---")
        # Record on default physics
        _record_single(str(model_path), args, tag="default")

        # Record on OOD conditions
        for ood in ["heavy_object", "low_friction", "combined_hard"]:
            old_ood = args.ood
            args.ood = ood
            _record_single(str(model_path), args, tag=ood)
            args.ood = old_ood


if __name__ == "__main__":
    main()
