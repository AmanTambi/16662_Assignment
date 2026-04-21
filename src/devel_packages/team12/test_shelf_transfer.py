"""
Shelf-to-shelf block transfer test using the team12 PRM.

Simulates picking a block off one shelf level and placing it on another.
Uses the same MuJoCo scene as Lab2 but queries the team12 PRM roadmap.

Run:
    cd team12
    python test_shelf_transfer.py
"""

import sys
import os
import numpy as np
import time

# ---- path setup: all heavy dependencies live in Lab2 ----
LAB2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Lab2')
sys.path.insert(0, LAB2)
# Change working directory so MuJoCo can find the XML / asset files
os.chdir(LAB2)

import mujoco as mj
from mujoco import viewer
import xml.etree.ElementTree as ET
import RobotUtil as rt
import Franka

# Import team12's path planner (uses team12_PRM.p, not Lab2's myPRM3.p)
sys.path.insert(0, os.path.join(LAB2, '..'))
import path_planner as team12_pp

# ---- scene constants (mirror Lab2/run_tasks.py) ----
EndofTable     = 0.55 + 0.135 + 0.05
R_SHELF_BOTTOM = 0.1375 + 0.005 + 0.005          # z of right-shelf bottom surface
R_SHELF_MIDDLE = R_SHELF_BOTTOM + 0.2
R_SHELF_TOP    = R_SHELF_BOTTOM + 0.4
R_SHELF_Z      = {0: R_SHELF_BOTTOM, 1: R_SHELF_MIDDLE, 2: R_SHELF_TOP}
L_SHELF_Z      = R_SHELF_BOTTOM                   # left shelf has only one level

BLOCK_HALF      = 0.02
GRASP_OFFSET    = 0.09
PRE_GRASP_DIST  = 0.10
TRANSIT_Z       = 0.75

GRIPPER_OPEN    = 0.04
GRIPPER_CLOSED  = 0.0125
HOME_CONFIG     = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
ARM_IDX         = [0, 1, 2, 3, 4, 5, 6]
GRIPPER_IDX     = 7

KP = np.array([120, 120, 100, 90, 60, 40, 30], dtype=float)
KD = np.array([  8,   8,   6,  5,  4,  3,  2], dtype=float)
SEGMENT_DURATION = 2.5
HOLD_DURATION    = 0.5

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


# ---- helpers ----

def Rz_local(angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.eye(4)
    R[0, 0] =  c; R[0, 1] = -s
    R[1, 0] =  s; R[1, 1] =  c
    return R


def side_grasp_pose(block_pos, side="+y"):
    bx, by, bz = block_pos
    if side == "-y":
        ee = [bx, by - GRASP_OFFSET, bz]
        T  = rt.rpyxyz2H([-np.pi/2, 0., 0.], ee) @ Rz_local(-np.pi/4)
        approach = np.array([0., 1., 0.])
    else:
        ee = [bx, by + GRASP_OFFSET, bz]
        T  = rt.rpyxyz2H([np.pi/2, 0., 0.], ee) @ Rz_local(-np.pi/4)
        approach = np.array([0., -1., 0.])
    return T, approach


def topdown_grasp_pose(block_pos):
    bx, by, bz = block_pos
    ee = [bx, by, bz + GRASP_OFFSET]
    T  = rt.rpyxyz2H([np.pi, 0., 0.], ee) @ Rz_local(np.pi/4)
    return T, np.array([0., 0., -1.])


def build_scene(block_start_pos):
    tree = ET.parse(ROOT_MODEL_XML)
    root = tree.getroot()

    visual = root.find("visual") or ET.SubElement(root, "visual")
    gl = visual.find("global") or ET.SubElement(visual, "global")
    gl.set("offwidth", "1280")
    gl.set("offheight", "960")

    for blk in BLOCKS:
        rt.add_free_block_to_model(
            tree=tree, name=blk[0], pos=blk[1],
            density=20, size=blk[2], rgba=[0.2, 0.2, 0.9, 1], free=False)

    rt.add_free_block_to_model(
        tree=tree, name="Block", pos=block_start_pos,
        density=20, size=[BLOCK_HALF]*3, rgba=[0., 0.9, 0.2, 1], free=True)

    tree.write(MODEL_XML, encoding="utf-8", xml_declaration=True)
    return MODEL_XML


# ---- MuJoCo control ----

def execute_trajectory(model, data, v, waypoints, gripper_cmd,
                       seg_dur=SEGMENT_DURATION, hold_dur=HOLD_DURATION):
    dt        = model.opt.timestep
    seg_steps = max(1, int(seg_dur / dt))
    hold_steps = int(hold_dur / dt)
    for i in range(len(waypoints) - 1):
        q_s, q_e = np.array(waypoints[i]), np.array(waypoints[i+1])
        t = 0.0
        for _ in range(seg_steps + hold_steps):
            q_des, qd_des = rt.interp_min_jerk(q_s, q_e, t, seg_dur)
            q  = data.qpos[ARM_IDX].copy()
            qd = data.qvel[ARM_IDX].copy()
            tau = KP*(q_des - q) + KD*(qd_des - qd)
            data.ctrl[ARM_IDX]     = tau + data.qfrc_bias[:7]
            data.ctrl[GRIPPER_IDX] = gripper_cmd
            mj.mj_step(model, data)
            v.sync()
            t += dt


def hold_at(model, data, v, q, gripper_cmd, duration=HOLD_DURATION):
    dt = model.opt.timestep
    for _ in range(int(duration / dt)):
        qc  = data.qpos[ARM_IDX].copy()
        qdc = data.qvel[ARM_IDX].copy()
        tau = KP*(q - qc) + KD*(0. - qdc)
        data.ctrl[ARM_IDX]     = tau + data.qfrc_bias[:7]
        data.ctrl[GRIPPER_IDX] = gripper_cmd
        mj.mj_step(model, data)
        v.sync()


def plan_and_exec(model, data, v, mybot, q_curr, T_target, gripper_cmd,
                  seg_dur=SEGMENT_DURATION, ik_seed=None):
    seed = ik_seed if ik_seed is not None else q_curr.copy()
    q_goal, err = mybot.IterInvKin(seed, T_target)
    if np.linalg.norm(err[:3]) > 0.005:
        print(f"  [WARN] IK position error: {np.linalg.norm(err[:3]):.4f} m")

    plan = team12_pp.PRMQuery(q_curr, q_goal)

    if plan is None:
        # PRM couldn't connect — fall back to direct linear interpolation in
        # joint space.  This is safe for short approach / retract moves where
        # the path is already known to be approximately clear (pre-grasp →
        # grasp, or grasp → pre-grasp).
        print("  [INFO] PRM returned None — using direct joint-space interpolation")
        n_steps = 5
        plan = [q_curr + (q_goal - q_curr) * t / n_steps
                for t in range(n_steps + 1)]

    execute_trajectory(model, data, v, plan, gripper_cmd, seg_dur)
    return np.array(q_goal)


def _grasp_pose_for_level(block_pos, level):
    """
    Choose a grasp strategy based on shelf height.
    Level 0 (bottom, z≈0.17m): top-down — keeps arm links above the table.
    Levels 1 & 2 (middle/top):  side (+y) — cleaner approach past shelf walls.
    """
    if level == 0:
        return topdown_grasp_pose(block_pos)
    return side_grasp_pose(block_pos, "+y")


def _transit_pose_for_level(xyz, level):
    """Transit (high-z waypoint) pose matching the grasp orientation."""
    xyz = list(xyz)
    xyz[2] = TRANSIT_Z
    if level == 0:
        return rt.rpyxyz2H([np.pi, 0., 0.], xyz) @ Rz_local(np.pi / 4)
    return rt.rpyxyz2H([np.pi / 2, 0., 0.], xyz) @ Rz_local(-np.pi / 4)


def pick_from_shelf(model, data, v, mybot, q, block_pos, level):
    T_grasp, approach = _grasp_pose_for_level(block_pos, level)
    pre = T_grasp.copy()
    pre[:3, 3] = T_grasp[:3, 3] - PRE_GRASP_DIST * approach
    q = plan_and_exec(model, data, v, mybot, q, pre, GRIPPER_OPEN)
    q = plan_and_exec(model, data, v, mybot, q, T_grasp, GRIPPER_OPEN, seg_dur=2.0)
    hold_at(model, data, v, q, GRIPPER_CLOSED, duration=2.0)
    q = plan_and_exec(model, data, v, mybot, q, pre, GRIPPER_CLOSED)
    return q


def place_on_shelf(model, data, v, mybot, q, block_pos, level):
    # small clearance so the block doesn't slam the shelf surface
    place_pos = list(block_pos)
    place_pos[2] += 0.03

    T_place, approach = _grasp_pose_for_level(place_pos, level)
    pre = T_place.copy()
    pre[:3, 3] = T_place[:3, 3] - PRE_GRASP_DIST * approach

    # arc over obstacles at a safe height before descending to pre-place
    T_transit = _transit_pose_for_level(pre[:3, 3], level)

    q = plan_and_exec(model, data, v, mybot, q, T_transit, GRIPPER_CLOSED)
    q = plan_and_exec(model, data, v, mybot, q, pre, GRIPPER_CLOSED)
    q = plan_and_exec(model, data, v, mybot, q, T_place, GRIPPER_CLOSED, seg_dur=2.0)
    hold_at(model, data, v, q, GRIPPER_OPEN, duration=1.0)
    q = plan_and_exec(model, data, v, mybot, q, pre, GRIPPER_OPEN)
    return q


def go_home(model, data, v, q):
    plan = team12_pp.PRMQuery(q, HOME_CONFIG)
    if plan is None:
        n = 8
        plan = [q + (HOME_CONFIG - np.array(q)) * t / n for t in range(n + 1)]
    execute_trajectory(model, data, v, plan, GRIPPER_OPEN)
    hold_at(model, data, v, HOME_CONFIG, GRIPPER_OPEN, duration=2.0)
    return HOME_CONFIG.copy()


# ---- test scenarios ----

TRANSFERS = [
    # (label, pick_level, pick_x, place_level, place_x)
    ("Bottom → Top shelf",   0, 0.50, 2, 0.40),
    ("Top → Middle shelf",   2, 0.40, 1, 0.50),
    ("Middle → Bottom shelf",1, 0.50, 0, 0.35),
]


def run_transfer(model, data, v, mybot, pick_level, pick_x, place_level, place_x):
    """
    Pick block from right shelf at pick_level/pick_x,
    place it at place_level/place_x.
    Grasp strategy is chosen per level (top-down for level 0, side for 1/2).
    """
    q = HOME_CONFIG.copy()

    pick_pos  = [pick_x,  -0.28, R_SHELF_Z[pick_level]  + BLOCK_HALF]
    place_pos = [place_x, -0.28, R_SHELF_Z[place_level] + BLOCK_HALF]

    print(f"\n  Pick  pos: {np.round(pick_pos, 3)}  (level {pick_level})")
    print(f"  Place pos: {np.round(place_pos, 3)}  (level {place_level})")

    print("\n--- Step 1: Hold at home ---")
    hold_at(model, data, v, HOME_CONFIG, GRIPPER_OPEN, duration=1.0)

    print("--- Step 2: Pick from shelf ---")
    q = pick_from_shelf(model, data, v, mybot, q, pick_pos, pick_level)

    print("--- Step 3: Place on shelf ---")
    q = place_on_shelf(model, data, v, mybot, q, place_pos, place_level)

    print("--- Step 4: Return home ---")
    q = go_home(model, data, v, q)
    return q


def init_block_on_shelf(level, x):
    """Start position: block resting on the right shelf at the given level."""
    return [x, -0.28, R_SHELF_Z[level] + BLOCK_HALF]


if __name__ == "__main__":
    np.random.seed(13)

    transfer_idx = 0           # change to 1 or 2 to test other scenarios
    label, pick_lvl, pick_x, place_lvl, place_x = TRANSFERS[transfer_idx]

    print(f"\n{'='*60}")
    print(f"  Shelf transfer test: {label}")
    print(f"{'='*60}")

    block_start = init_block_on_shelf(pick_lvl, pick_x)
    xml_path    = build_scene(block_start)
    model       = mj.MjModel.from_xml_path(xml_path)
    data        = mj.MjData(model)
    data.qpos[ARM_IDX] = HOME_CONFIG
    data.qvel[:]       = 0
    mj.mj_forward(model, data)

    mybot = Franka.FrankArm()

    v = viewer.launch_passive(model, data)
    v.cam.distance  = 2.5
    v.cam.azimuth   = 135
    v.cam.elevation = -25
    v.cam.lookat[:] = [0.3, 0.0, 0.3]

    print("Waiting for viewer to load...")
    time.sleep(10.0)
    hold_at(model, data, v, HOME_CONFIG, GRIPPER_OPEN, duration=2.0)

    try:
        run_transfer(model, data, v, mybot, pick_lvl, pick_x, place_lvl, place_x)
        print("\nTransfer complete. Press Ctrl+C to exit.")
        while v.is_running():
            mj.mj_step(model, data)
            v.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        v.close()
