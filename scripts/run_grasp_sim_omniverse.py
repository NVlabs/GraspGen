# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Isaac Sim script: after opening box_with_grasps_sim.usd, run this so that on
Play the grippers close and grasp the object.

Usage in Isaac Sim:
  1. Open the stage: File > Open > box_with_grasps_sim.usd
  2. Open Script Editor (Window > Script Editor)
  3. Run this script (e.g. paste and Run, or run as standalone with isaacsim python)
  4. Press Play – grippers will move to closed position after a short delay.

Requires: Isaac Sim with omni.isaac.core (bundled).
"""

import numpy as np

# Isaac Sim / Omniverse imports (available when run inside Isaac Sim)
try:
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.types import ArticulationAction
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

# Delay after play (seconds) before commanding grippers to close
GRASP_CLOSE_DELAY_S = 0.5
# For Robotiq-style grippers: use lower joint limit as "closed" (fingers close when going to min).
# Set to "upper" if your gripper closes at the upper limit.
CLOSED_JOINT_LIMIT = "lower"


def _get_articulation_roots_under_world(stage):
    """Return prim paths of articulation roots under /World (Env_*/gripper)."""
    roots = []
    world = stage.GetPrimAtPath("/World")
    if not world:
        return roots
    for env in world.GetChildren():
        if not env.GetName().startswith("Env_"):
            continue
        gripper_path = env.GetPath().pathString + "/gripper"
        gripper = stage.GetPrimAtPath(gripper_path)
        if gripper and gripper.IsValid():
            # Referenced gripper root may have ArticulationRootAPI; if not, use path anyway for Articulation()
            roots.append(gripper_path)
    return roots


def _get_closed_joint_positions(articulation, use_lower=True):
    """Get joint positions for 'closed' grasp from articulation limits."""
    joint_names = articulation.dof_names
    limits = articulation.get_joint_limits()
    if limits is None or len(limits) != len(joint_names):
        return None
    pos = []
    for (low, high) in limits:
        if use_lower:
            pos.append(float(low))
        else:
            pos.append(float(high))
    return np.array(pos, dtype=np.float64)


def run_grasp_on_play():
    """Register a timeline callback so that after Play, grippers close."""
    if not ISAAC_AVAILABLE:
        print("run_grasp_sim_omniverse: omni.isaac.core not available. Run this script inside Isaac Sim.")
        return

    from omni.isaac.core.utils.stage import get_current_stage
    import carb

    stage = get_current_stage()
    if not stage:
        print("run_grasp_sim_omniverse: No stage open. Open box_with_grasps_sim.usd first.")
        return

    root_paths = _get_articulation_roots_under_world(stage)
    if not root_paths:
        print("run_grasp_sim_omniverse: No /World/Env_*/gripper articulation roots found.")
        return

    world = World.instance()
    if world is None:
        print("run_grasp_sim_omniverse: World not initialized. Press Play once, then run this script again, or use the callback below.")
        return

    articulations = []
    for path in root_paths:
        art = world.scene.get(path)
        if art is None:
            art = Articulation(path)
            world.scene.add(art)
        articulations.append(art)

    # After a short delay, set all grippers to closed
    def _close_grippers(dt):
        for art in articulations:
            if art is None:
                continue
            try:
                closed_pos = _get_closed_joint_positions(art, use_lower=(CLOSED_JOINT_LIMIT == "lower"))
                if closed_pos is not None:
                    art.apply_action(ArticulationAction(joint_positions=closed_pos))
            except Exception as e:
                carb.log_warn(f"run_grasp_sim_omniverse: {e}")

    # Single-shot after delay: use a simple timer via world's update
    elapsed = [0.0]

    def _on_timestep(timestep):
        elapsed[0] += timestep
        if elapsed[0] >= GRASP_CLOSE_DELAY_S:
            _close_grippers(timestep)
            world.remove_timestep_callback("grasp_close")

    world.add_timestep_callback("grasp_close", _on_timestep)
    print("run_grasp_sim_omniverse: On Play, grippers will close after", GRASP_CLOSE_DELAY_S, "s.")


def main():
    """Entry when run as script. Assumes Isaac Sim has already loaded the stage."""
    run_grasp_on_play()


if __name__ == "__main__":
    main()
