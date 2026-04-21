"""
team12/run_tasks.py

Same pick-place pipeline as Lab2/run_tasks.py, but queries the team12 PRM
(team12_PRM.p) instead of Lab2's myPRM3.p.

Task sequence (identical to Lab2):
  1. Pick green block off table (top-down grasp)
  2. Place on left shelf (top-down)
  3. Re-grasp from left shelf (side, -y)
  4. Transit high, place on right shelf (side, +y)
  5. Return home

Run with:
    cd team12
    mjpython run_tasks.py
"""

import sys
import os
import numpy as np
import time
import xml.etree.ElementTree as ET

# ---- path setup ----
TEAM12 = os.path.dirname(os.path.abspath(__file__))
LAB2   = os.path.join(TEAM12, 'Lab2')

# team12 must come FIRST so that `import path_planner` resolves to
# team12/path_planner.py (which uses team12_PRM.p) rather than Lab2's copy
sys.path.insert(0, TEAM12)
sys.path.insert(1, LAB2)

# chdir so MuJoCo resolves relative XML/asset paths from Lab2
os.chdir(LAB2)

import mujoco as mj
from mujoco import viewer
import RobotUtil as rt
import Franka

# motion_utils imports `path_planner` — because TEAM12 is first in sys.path
# it will import team12/path_planner.py, which queries team12_PRM.p
from motion_utils import (
    hold_at_config, plan_and_execute,
    pick_up, drop, home,
    HOME_CONFIG, GRIPPER_OPEN, GRIPPER_CLOSED, ARM_IDX,
)

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

ROOT_MODEL_XML = "franka_emika_panda/panda_torque_table.xml"
MODEL_XML      = "franka_emika_panda/panda_torque_table_shelves.xml"

BLOCKS = [
    ["TablePlane",        [EndofTable-0.275, 0., -0.005],                                [0.275, 0.504, 0.0051]],
    ["LShelfDistal",      [EndofTable-0.09-0.0225, 0.504-0.045-0.0225, 0.315],           [0.0225, 0.0225, 0.315]],
    ["LShelfProximal",    [EndofTable-0.55-0.0225, 0.504-0.045-0.0225, 0.3825-0.135],    [0.0225, 0.0225, 0.3825]],
    ["LShelfBack",        [EndofTable-0.55-0.0225-0.09, 0.504-0.045-0.0225, 0.3825-0.135], [0.0225, 0.0225, 0.3825]],
    ["LShelfMid",         [EndofTable-0.32, 0.504-0.045-0.0225, 0.315],                  [0.0225, 0.0225, 0.315]],
    ["LShelfArch",        [EndofTable-0.275-0.135+0.0225, 0.504-0.045-0.0225, 0.63+0.0225], [0.315, 0.0225, 0.0225]],
    ["LShelfBottom",      [EndofTable-0.275-0.135+0.0225, 0.504-0.09-0.135/2., 0.1375+0.005], [0.2525, 0.135/2., 0.005]],
    ["LShelfBottomSupp1", [EndofTable-0.55-0.0225-0.09+0.045, 0.504-0.225/2., 0.1375-0.0225], [0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSupp2", [EndofTable-0.32-0.045, 0.504-0.225/2., 0.1375-0.0225],        [0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSupp3", [EndofTable-0.09-0.0225-0.045, 0.504-0.225/2., 0.1375-0.0225], [0.0225, 0.1125, 0.0225]],
    ["LShelfBottomSuppB", [EndofTable-0.275-0.135+0.0225, 0.504-0.0225, 0.1375+0.0225],  [0.315, 0.0225, 0.0225]],
    ["RShelfDistal",      [EndofTable-0.09-0.0225, -0.504+0.045+0.0225, 0.315],          [0.0225, 0.0225, 0.315]],
    ["RShelfProximal",    [EndofTable-0.55-0.0225, -0.504+0.045+0.0225, 0.3825-0.135],   [0.0225, 0.0225, 0.3825]],
    ["RShelfBack",        [EndofTable-0.55-0.0225-0.09, -0.504+0.045+0.0225, 0.3825-0.135], [0.0225, 0.0225, 0.3825]],
    ["RShelfMid",         [EndofTable-0.32, -0.504+0.045+0.0225, 0.315],                 [0.0225, 0.0225, 0.315]],
    ["RShelfArch",        [EndofTable-0.275-0.135+0.0225, -0.504+0.045+0.0225, 0.63+0.0225], [0.315, 0.0225, 0.0225]],
    ["RShelfBottom",      [EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005], [0.2525, 0.135/2., 0.005]],
    ["RShelfBottomSupp1", [EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225], [0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSupp2", [EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225],       [0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSupp3", [EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225], [0.0225, 0.1125, 0.0225]],
    ["RShelfBottomSuppB", [EndofTable-0.275-0.135+0.0225, -0.504+0.0225, 0.1375+0.0225], [0.315, 0.0225, 0.0225]],
    ["RShelfMiddle",      [EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.2], [0.2525, 0.135/2., 0.005]],
    ["RShelfMiddleSupp1", [EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.2], [0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSupp2", [EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.2],    [0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSupp3", [EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.2], [0.0225, 0.1125, 0.0225]],
    ["RShelfMiddleSuppB", [EndofTable-0.275-0.135+0.0225, -0.504+0.0225, 0.1375+0.0225+.2], [0.315, 0.0225, 0.0225]],
    ["RShelfTop",         [EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.4], [0.2525, 0.135/2., 0.005]],
    ["RShelfTopSupp1",    [EndofTable-0.55-0.0225-0.09+0.045, -0.504+0.225/2., 0.1375-0.0225+.4], [0.0225, 0.1125, 0.0225]],
    ["RShelfTopSupp2",    [EndofTable-0.32-0.045, -0.504+0.225/2., 0.1375-0.0225+.4],    [0.0225, 0.1125, 0.0225]],
    ["RShelfTopSupp3",    [EndofTable-0.09-0.0225-0.045, -0.504+0.225/2., 0.1375-0.0225+.4], [0.0225, 0.1125, 0.0225]],
    ["RShelfTopSuppB",    [EndofTable-0.275-0.135+0.0225, -0.504+0.0225, 0.1375+0.0225+.4], [0.315, 0.0225, 0.0225]],
]


def Rz_local(angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4)
    R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
    return R


def build_scene():
    """Rebuild scene XML with correct body names (RobotUtil fix applied)."""
    tree = ET.parse(ROOT_MODEL_XML)
    root = tree.getroot()

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    gl = visual.find("global")
    if gl is None:
        gl = ET.SubElement(visual, "global")
    gl.set("offwidth", "1280")
    gl.set("offheight", "960")

    for blk in BLOCKS:
        rt.add_free_block_to_model(
            tree=tree, name=blk[0], pos=blk[1],
            density=20, size=blk[2], rgba=[0.2, 0.2, 0.9, 1], free=False)

    rt.add_free_block_to_model(
        tree=tree, name="Block", pos=BLOCK_START_POS,
        density=20, size=[BLOCK_HALF]*3,
        rgba=[0.0, 0.9, 0.2, 1], free=True)

    tree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)
    return MODEL_XML


def topdown_grasp_pose(block_pos):
    bx, by, bz = block_pos
    T = rt.rpyxyz2H([np.pi, 0., 0.], [bx, by, bz + GRASP_OFFSET]) @ Rz_local(np.pi/4)
    return T, np.array([0., 0., -1.])


def side_grasp_pose(block_pos, side="+y"):
    bx, by, bz = block_pos
    if side == "-y":
        T = rt.rpyxyz2H([-np.pi/2, 0., 0.], [bx, by - GRASP_OFFSET, bz]) @ Rz_local(-np.pi/4)
        approach = np.array([0., 1., 0.])
    else:
        T = rt.rpyxyz2H([np.pi/2, 0., 0.], [bx, by + GRASP_OFFSET, bz]) @ Rz_local(-np.pi/4)
        approach = np.array([0., -1., 0.])
    return T, approach


def left_shelf_side_grasp_pose():
    block_pos = [0.50, 0.27, L_SHELF_SURFACE_Z + BLOCK_HALF]
    return side_grasp_pose(block_pos, side="-y")


def right_shelf_place_pose(level=0, x=0.50):
    block_pos = [x, -0.28, R_SHELF_Z[level] + BLOCK_HALF + 0.04]
    return side_grasp_pose(block_pos, side="+y")


# ---- Named placement targets (from calibrated poses in poses.txt) ----
# x: "left" = closer to robot base (higher x), "right" = further (lower x)
# level: top=2, mid=1
PLACE_TARGETS = {
    "top_left":  (2, 0.56),
    "top_right": (2, 0.30),
    "mid_left":  (1, 0.56),
    "mid_right": (1, 0.30),
}


def run_task(model, data, v, mybot, target="top_left"):
    """
    Pick block from table, stage on left shelf, regrasp, place on right shelf.

    target: one of "top_left", "top_right", "mid_left", "mid_right"
    """
    if target not in PLACE_TARGETS:
        raise ValueError(f"Unknown target '{target}'. Choose from: {list(PLACE_TARGETS)}")

    shelf_level, shelf_x = PLACE_TARGETS[target]
    q = HOME_CONFIG.copy()

    print("\n=== Step 1: Pick block from table ===")
    T_grasp, approach = topdown_grasp_pose(BLOCK_START_POS)
    q = pick_up(model, data, v, mybot, q, T_grasp, approach)

    print("\n=== Step 2: Place on left shelf (staging) ===")
    place_pos = [0.50, 0.29, L_SHELF_SURFACE_Z + 0.075]
    T_lplace, l_approach = topdown_grasp_pose(place_pos)
    q = drop(model, data, v, mybot, q, T_lplace, l_approach)

    print("\n=== Step 3: Re-grasp from left shelf (side) ===")
    T_regrasp, r_approach = left_shelf_side_grasp_pose()
    q = pick_up(model, data, v, mybot, q, T_regrasp, r_approach)

    print(f"\n=== Step 4: Place → {target}  (level={shelf_level}, x={shelf_x}) ===")
    T_rplace, rp_approach = right_shelf_place_pose(shelf_level, x=shelf_x)
    pre_xyz = T_rplace[:3, 3] - PRE_GRASP_DIST * rp_approach
    T_rtransit = rt.rpyxyz2H([np.pi/2, 0., 0.],
                              [pre_xyz[0], pre_xyz[1], TRANSIT_Z]) @ Rz_local(-np.pi/4)
    q = plan_and_execute(model, data, v, mybot, q, T_rtransit, GRIPPER_CLOSED)
    q = drop(model, data, v, mybot, q, T_rplace, rp_approach)

    print("\n=== Step 5: Safe transit before home ===")
    # Rise to a neutral high pose above the workspace centre before going home.
    # This prevents the arm from swinging through the table on the way back.
    T_safe = rt.rpyxyz2H([np.pi, 0., 0.], [0.4, 0.0, 0.55]) @ Rz_local(np.pi / 4)
    q = plan_and_execute(model, data, v, mybot, q, T_safe, GRIPPER_OPEN)

    print("\n=== Step 6: Return home ===")
    home(model, data, v, q)


def reload_sim(arm_config):
    """Build a fresh sim — called between runs to reset block position."""
    xml_path = build_scene()
    model = mj.MjModel.from_xml_path(xml_path)
    data  = mj.MjData(model)
    data.qpos[ARM_IDX] = arm_config
    data.qvel[:]       = 0
    mj.mj_forward(model, data)
    return model, data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="team12 shelf placement demo")
    parser.add_argument(
        "targets",
        nargs="*",
        default=list(PLACE_TARGETS.keys()),
        help=f"One or more placement targets. Choices: {list(PLACE_TARGETS.keys())}. "
             "Defaults to all four in sequence.",
    )
    args = parser.parse_args()

    for t in args.targets:
        if t not in PLACE_TARGETS:
            parser.error(f"Unknown target '{t}'. Choose from: {list(PLACE_TARGETS.keys())}")

    np.random.seed(13)

    model, data = reload_sim(HOME_CONFIG)
    mybot = Franka.FrankArm()

    v = viewer.launch_passive(model, data)
    v.cam.distance  = 2.5
    v.cam.azimuth   = 135
    v.cam.elevation = -25
    v.cam.lookat[:] = [0.3, 0.0, 0.3]

    print(f"Placement sequence: {args.targets}")
    print("Waiting for viewer...")
    time.sleep(15.0)
    hold_at_config(model, data, v, HOME_CONFIG, GRIPPER_OPEN, duration=2.0)

    try:
        for run_idx, target in enumerate(args.targets):
            print(f"\n{'='*50}")
            print(f"  RUN {run_idx+1}/{len(args.targets)}: target = {target}")
            print(f"{'='*50}")

            if run_idx > 0:
                v.close()
                model, data = reload_sim(HOME_CONFIG)
                v = viewer.launch_passive(model, data)
                v.cam.distance  = 2.5
                v.cam.azimuth   = 135
                v.cam.elevation = -25
                v.cam.lookat[:] = [0.3, 0.0, 0.3]
                time.sleep(5.0)
                hold_at_config(model, data, v, HOME_CONFIG, GRIPPER_OPEN, duration=2.0)

            run_task(model, data, v, mybot, target=target)

        print(f"\nAll {len(args.targets)} run(s) complete. Holding at home. Press Ctrl+C to exit.")
        # Actively hold the arm at HOME_CONFIG with PD control so it doesn't
        # sag under gravity while the viewer is still open.
        while v.is_running():
            hold_at_config(model, data, v, HOME_CONFIG, GRIPPER_OPEN, duration=1.0)

    except KeyboardInterrupt:
        pass
    finally:
        v.close()
