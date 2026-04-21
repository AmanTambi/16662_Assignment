"""
run_tasks_frankapy.py

Physical execution pipeline using frankapy. 
Replaces MuJoCo simulation and custom PRM tracking with Franka Emika Panda hardware commands.
"""

import sys
import numpy as np
import time
import argparse
from scipy.spatial.transform import Rotation as R_sci
from poses import *

# frankapy imports
from frankapy import FrankaArm
from autolab_core import RigidTransform

# ---- scene constants ----
EndofTable       = 0.55 + 0.135 + 0.05
BLOCK_START_POS  = [EndofTable - 0.145, 0.0, 0.05]

L_SHELF_SURFACE_Z = 0.1375 + 0.005 + 0.005
R_SHELF_BOTTOM_Z  = 0.1375 + 0.005 + 0.005
R_SHELF_MIDDLE_Z  = R_SHELF_BOTTOM_Z + 0.2
R_SHELF_TOP_Z     = R_SHELF_BOTTOM_Z + 0.4
R_SHELF_Z         = {0: R_SHELF_BOTTOM_Z, 1: R_SHELF_MIDDLE_Z, 2: R_SHELF_TOP_Z}

BLOCK_HALF     = 0.02
PRE_GRASP_DIST = 0.10
TRANSIT_Z      = 0.70
GRASP_OFFSET   = 0.09

# Named placement targets
PLACE_TARGETS = {
    "top_left":  (2, 0.56),
    "top_right": (2, 0.30),
    "mid_left":  (1, 0.56),
    "mid_right": (1, 0.30),
}


# ---- Math & Transform Utilities ----
def rpyxyz2H(rpy, xyz):
    """Replaces RobotUtil.rpyxyz2H using scipy."""
    R = R_sci.from_euler('xyz', rpy).as_matrix()
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = xyz
    return H

def Rz_local(angle):
    """Local Z rotation matrix."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4)
    R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
    return R

def numpy_to_rigid_transform(T):
    """Converts a 4x4 numpy SE(3) matrix to autolab_core RigidTransform for frankapy."""
    return RigidTransform(
        rotation=T[:3, :3],
        translation=T[:3, 3],
        from_frame='franka_tool',
        to_frame='world'
    )

# ---- Pose Generators ----
def topdown_grasp_pose(block_pos):
    bx, by, bz = block_pos
    T = rpyxyz2H([np.pi, 0., 0.], [bx, by, bz + GRASP_OFFSET]) @ Rz_local(0)
    return T, np.array([0., 0., -1.])

def side_grasp_pose(block_pos, side="+y"):
    bx, by, bz = block_pos
    if side == "-y":
        T = rpyxyz2H([-np.pi/2, 0., 0.], [bx, by - GRASP_OFFSET, bz]) @ Rz_local(-np.pi/2)
        approach = np.array([0., 1., 0.])
    else:
        T = rpyxyz2H([np.pi/2, 0., 0.], [bx, by + GRASP_OFFSET, bz]) @ Rz_local(np.pi/2)
        approach = np.array([0., -1., 0.])
    return T, approach

def left_shelf_side_grasp_pose():
    block_pos = [0.50, 0.27, L_SHELF_SURFACE_Z + BLOCK_HALF]
    return side_grasp_pose(block_pos, side="-y")

def right_shelf_place_pose(level=0, x=0.50):
    block_pos = [x, -0.28, R_SHELF_Z[level] + BLOCK_HALF + 0.04]
    return side_grasp_pose(block_pos, side="+y")


# ---- Franka Hardware Primitives ----
def execute_pick(fa, T_grasp, approach_vec):
    """Approaches, grabs, and retreats using frankapy commands."""
    T_pre = T_grasp.copy()
    T_pre[:3, 3] -= PRE_GRASP_DIST * approach_vec
    
    fa.goto_pose(numpy_to_rigid_transform(T_pre), duration=3.0)
    fa.open_gripper()
    
    fa.goto_pose(numpy_to_rigid_transform(T_grasp), duration=2.0)
    fa.close_gripper()
    time.sleep(0.5) # Wait for grasp to secure
    
    fa.goto_pose(numpy_to_rigid_transform(T_pre), duration=2.0)

def execute_drop(fa, T_place, approach_vec):
    """Approaches, drops, and retreats using frankapy commands."""
    T_pre = T_place.copy()
    T_pre[:3, 3] -= PRE_GRASP_DIST * approach_vec
    
    fa.goto_pose(numpy_to_rigid_transform(T_pre), duration=3.0)
    fa.goto_pose(numpy_to_rigid_transform(T_place), duration=2.0)
    
    fa.open_gripper()
    time.sleep(0.5) # Wait for drop to clear
    
    fa.goto_pose(numpy_to_rigid_transform(T_pre), duration=2.0)


# ---- Main Task Loop ----
def run_task(fa, target="top_left"):
    if target not in PLACE_TARGETS:
        raise ValueError(f"Unknown target '{target}'")

    shelf_level, shelf_x = PLACE_TARGETS[target]

    print("\n=== Step 1: Pick block from table ===")
    T_grasp, approach = topdown_grasp_pose(BLOCK_START_POS)
    execute_pick(fa, T_grasp, approach)
    
    T_lplace, l_approach = topdown_grasp_pose(FRANKA_POSES["camera_view"]["Tra"])

    print("\n=== Step 2: Place on left shelf (staging) ===")
    place_pos = [0.50, 0.29, L_SHELF_SURFACE_Z + 0.075]
    T_lplace, l_approach = topdown_grasp_pose(place_pos)
    execute_drop(fa, T_lplace, l_approach)

    print("\n=== Step 3: Re-grasp from left shelf (side) ===")
    T_regrasp, r_approach = left_shelf_side_grasp_pose()
    
    execute_pick(fa, T_regrasp, r_approach)

    print(f"\n=== Step 4: Place → {target}  (level={shelf_level}, x={shelf_x}) ===")
    T_rplace, rp_approach = right_shelf_place_pose(shelf_level, x=shelf_x)
    
    # Transit High
    pre_xyz = T_rplace[:3, 3] - PRE_GRASP_DIST * rp_approach
    T_rtransit = rpyxyz2H([np.pi/2, 0., 0.], [pre_xyz[0], pre_xyz[1], TRANSIT_Z]) @ Rz_local(-np.pi/4)
    
    fa.goto_pose(numpy_to_rigid_transform(T_rtransit), duration=3.0)
    execute_drop(fa, T_rplace, rp_approach)

    print("\n=== Step 5: Safe transit before home ===")
    T_safe = rpyxyz2H([np.pi, 0., 0.], [0.4, 0.0, 0.55]) @ Rz_local(np.pi / 4)
    fa.goto_pose(numpy_to_rigid_transform(T_safe), duration=3.0)

    print("\n=== Step 6: Return home ===")
    fa.reset_joints()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physical frankapy shelf placement demo")
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(PLACE_TARGETS.keys()),
        help=f"Placement targets. Choices: {list(PLACE_TARGETS.keys())}.",
    )
    args = parser.parse_args()

    # Initialize physical robot communication
    print("Initializing FrankaArm. Ensure robot is unlocked and in FCM mode...")
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()

    print(f"Placement sequence: {args.targets}")
    time.sleep(2.0)

    try:
        for run_idx, target in enumerate(args.targets):
            print(f"\n{'='*50}")
            print(f"  RUN {run_idx+1}/{len(args.targets)}: target = {target}")
            print(f"{'='*50}")

            if run_idx > 0:
                print("Returning home to reset for next run...")
                fa.reset_joints()
                time.sleep(2.0) # Pause so you can manually reset the block

            run_task(fa, target=target)

        print(f"\nAll {len(args.targets)} run(s) complete.")
        fa.reset_joints()

    except KeyboardInterrupt:
        print("\nStopping physical robot execution safely.")
        # Stops the current motion if ctrl+c is pressed
        fa.stop_skill()