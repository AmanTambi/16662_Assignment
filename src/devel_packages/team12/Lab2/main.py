import os
import numpy as np
import time

# FrankaPy imports
from frankapy import FrankaArm
from autolab_core import RigidTransform

# Custom planner and robot utilities
import RobotUtil as rt
import Franka
from planner import build_obstacle_model, build_prm, make_prm_plan_fn

# Import the poses we generated earlier
from poses import FRANKA_POSES

# --- CONSTANTS & CONFIGURATION ---
EndofTable = 0.55 + 0.135 + 0.05
TRAJ_FILE = "trajectories.npz"
PRM_FILE = "prm.npz"

BLOCKS = [
    ["TablePlane", [EndofTable-0.275, 0., -0.005], [0.275, 0.504, 0.0051]],
    ["LShelfDistal", [EndofTable-0.09-0.0225, 0.504-0.045-0.0225, 0.315], [0.0225, 0.0225, 0.315]],
    ["LShelfProximal", [EndofTable-0.55-0.0225, 0.504-0.045-0.0225, 0.3825-0.135], [0.0225, 0.0225, 0.3825]],
    ["LShelfBack", [EndofTable-0.55-0.0225-0.09, 0.504-0.045-0.0225, 0.3825-0.135], [0.0225, 0.0225, 0.3825]],
    ["LShelfMid", [EndofTable-0.32, 0.504-0.045-0.0225, 0.315], [0.0225, 0.0225, 0.315]],
    ["LShelfArch", [EndofTable-0.275-0.135+0.0225, 0.504-0.045-0.0225, 0.63+0.0225], [0.315, 0.0225, 0.0225]],
    ["LShelfBottom", [EndofTable-0.275-0.135+0.0225, 0.504-0.09-0.135/2., 0.1375+0.005], [0.2525, 0.135/2., 0.005]],
    ["RShelfDistal", [EndofTable-0.09-0.0225, -0.504+0.045+0.0225, 0.315], [0.0225, 0.0225, 0.315]],
    ["RShelfProximal", [EndofTable-0.55-0.0225, -0.504+0.045+0.0225, 0.3825-0.135], [0.0225, 0.0225, 0.3825]],
    ["RShelfBack", [EndofTable-0.55-0.0225-0.09, -0.504+0.045+0.0225, 0.3825-0.135], [0.0225, 0.0225, 0.3825]],
    ["RShelfMid", [EndofTable-0.32, -0.504+0.045+0.0225, 0.315], [0.0225, 0.0225, 0.315]],
    ["RShelfArch", [EndofTable-0.275-0.135+0.0225, -0.504+0.045+0.0225, 0.63+0.0225], [0.315, 0.0225, 0.0225]],
    ["RShelfBottom", [EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005], [0.2525, 0.135/2., 0.005]],
    ["RShelfMiddle", [EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.2], [0.2525, 0.135/2., 0.005]],
    ["RShelfTop", [EndofTable-0.275-0.135+0.0225, -0.504+0.09+0.135/2., 0.1375+0.005+.4], [0.2525, 0.135/2., 0.005]],
]

HOME_Q = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]

# --- HELPER FUNCTIONS ---
def build_T_matrix(pose_dict):
    """Converts a dictionary with 'Tra' and 'Rot' into a 4x4 Homogeneous Transform"""
    T = np.eye(4)
    T[:3, :3] = pose_dict["Rot"]
    T[:3, 3] = pose_dict["Tra"]
    return T

def matrix_to_rigid_transform(T):
    """Converts 4x4 matrix to autolab_core RigidTransform for fa.goto_pose"""
    return RigidTransform(rotation=T[:3, :3], translation=T[:3, 3], from_frame='franka_tool', to_frame='world')

def exec_path(fa, path, duration_per_step=10.0):
    """Executes a list of joint configurations sequentially"""
    if path is None or len(path) == 0:
        print("Warning: Path is empty or planning failed!")
        return
    for i, q in enumerate(path):
        print(f"    -> Moving to waypoint {i+1}/{len(path)}")
        fa.goto_joints(q, duration=duration_per_step, ignore_virtual_walls=True)

def plan_or_load(plan_fn, current_q, target_q, task_name, trajectories_dict):
    """Checks for cached trajectory; if missing, plans via PRM and saves it."""
    if task_name in trajectories_dict:
        print(f"  [Cache HIT] Using saved trajectory for '{task_name}'")
        return trajectories_dict[task_name]
    
    print(f"  [Cache MISS] Running PRM to plan path for '{task_name}'...")
    path = plan_fn(current_q, target_q, task_name)
    
    # Save the new path to our dictionary and immediately write to disk
    trajectories_dict[task_name] = path
    np.savez(TRAJ_FILE, **trajectories_dict)
    print(f"  [Saved] Trajectory '{task_name}' saved to {TRAJ_FILE}.")
    
    return path

# --- MAIN BEHAVIOR FUNCTIONS ---
def pick_and_show(fa, robot, plan_fn, pick_pose_name, current_loc_name, trajectories, enable_collision=True):
    print(f"\nExecuting pick_and_show (Collision Avoidance: {enable_collision})")
    
    pick_pose_dict = FRANKA_POSES[pick_pose_name]
    cam_pose_name = "show_object_in_camera"
    
    current_q = list(fa.get_joints())
    q_pick = pick_pose_dict["joints"]
    q_cam = FRANKA_POSES[cam_pose_name]["joints"]

    fa.open_gripper()

    if enable_collision:
        # 1. Path to Pick
        task_name = f"{current_loc_name}_to_{pick_pose_name}"
        path_to_pick = plan_or_load(plan_fn, current_q, q_pick, task_name, trajectories)
        exec_path(fa, path_to_pick)
    else:
        # 1. fa.goto_pose with safety pre-approach
        print("  Using fa.goto_pose with safety pre-approach...")
        T_pick = build_T_matrix(pick_pose_dict)
        T_above_pick = np.copy(T_pick)
        T_above_pick[2, 3] += 0.10
        fa.goto_pose(matrix_to_rigid_transform(T_above_pick), duration=3.0, ignore_virtual_walls=True)
        fa.goto_pose(matrix_to_rigid_transform(T_pick), duration=2.0, ignore_virtual_walls=True)

    # 2. Grip
    print("  Gripping...")
    fa.close_gripper()
    time.sleep(1.0)

    # 3. Move to Camera
    current_q = list(fa.get_joints())
    if enable_collision:
        task_name = f"{pick_pose_name}_to_{cam_pose_name}"
        path_to_cam = plan_or_load(plan_fn, current_q, q_cam, task_name, trajectories)
        exec_path(fa, path_to_cam)
    else:
        print("  Directly going to camera pose via goto_pose...")
        T_cam = build_T_matrix(FRANKA_POSES[cam_pose_name])
        fa.goto_pose(matrix_to_rigid_transform(T_cam), duration=3.0, ignore_virtual_walls=True)

    return cam_pose_name


def drop_to_shelf(fa, robot, plan_fn, shelf_pose_name, current_loc_name, trajectories, enable_collision=True):
    print(f"\nExecuting drop_to_shelf (Collision Avoidance: {enable_collision})")
    
    shelf_pose_dict = FRANKA_POSES[shelf_pose_name]
    current_q = list(fa.get_joints())
    q_drop = shelf_pose_dict["joints"]

    if enable_collision:
        task_name = f"{current_loc_name}_to_{shelf_pose_name}"
        path_to_drop = plan_or_load(plan_fn, current_q, q_drop, task_name, trajectories)
        exec_path(fa, path_to_drop)
    else:
        print("  Using fa.goto_pose for shelf approach...")
        T_drop = build_T_matrix(shelf_pose_dict)
        fa.goto_pose(matrix_to_rigid_transform(T_drop), duration=3.0, ignore_virtual_walls=True)
    
    print("  Releasing object...")
    fa.open_gripper()
    time.sleep(1.0)
    
    return shelf_pose_name


if __name__ == "__main__":
    # --- FLAG CONFIGURATION ---
    enable_collision = True 
    # --------------------------

    print("Initializing FrankaArm...")
    fa = FrankaArm()
    fa.reset_joints()
    fa.open_gripper()

    # Explicitly move to the designated HOME_Q before registering 'home'
    print("Moving explicitly to HOME_Q configuration...")
    fa.goto_joints(HOME_Q, duration=3.0, ignore_virtual_walls=True)

    robot = Franka.FrankArm()
    plan_fn = None
    trajectories = {}
    roadmap = {}
    current_location = "home" # Safe to set this now that we are physically there

    if enable_collision:
        # Load trajectories if the file exists
        if os.path.exists(TRAJ_FILE):
            print(f"Found {TRAJ_FILE}. Loading cached trajectories...")
            with np.load(TRAJ_FILE, allow_pickle=True) as data:
                # Convert loaded numpy arrays back to standard Python lists
                for key in data.files:
                    trajectories[key] = data[key].tolist() 
        else:
            print(f"No cache found. {TRAJ_FILE} will be created.")

        print("Building Obstacle Model and PRM...")
        pointsObs, axesObs = build_obstacle_model(BLOCKS)
        if os.path.exists(PRM_FILE):
            print(f"Found {PRM_FILE}. Loading cached prm model...")
            with np.load(PRM_FILE, allow_pickle=True) as data:
                # Convert loaded numpy arrays back to standard Python lists
                for key in data.files:
                    roadmap[key] = data[key].tolist() 
        else:
            roadmap = build_prm(robot, pointsObs, axesObs, n_vertices=1000, k_neighbors=10)
            np.savez(PRM_FILE, **roadmap)

        plan_fn = make_prm_plan_fn(roadmap, robot, pointsObs, axesObs)
    else:
        print("Collision Avoidance OFF. Using Cartesian moves.")

    try:
        # 1. Pick and show
        target_pick = "pick_object"
        current_location = pick_and_show(fa, robot, plan_fn, target_pick, current_location, trajectories, enable_collision)
        
        time.sleep(1.0)

        # 2. Drop to shelf
        target_drop = "middle_shelf_middle"
        current_location = drop_to_shelf(fa, robot, plan_fn, target_drop, current_location, trajectories, enable_collision)

        print("\nSequence complete. Returning home.")
        current_q = list(fa.get_joints())
        if enable_collision:
            task_name = f"{current_location}_to_home"
            path_to_home = plan_or_load(plan_fn, current_q, HOME_Q, task_name, trajectories)
            exec_path(fa, path_to_home)
        else:
            fa.goto_joints(HOME_Q, duration=3.0, ignore_virtual_walls=True)
            
        current_location = "home"

    except KeyboardInterrupt:
        print("\nStopped by user.")
        fa.stop_skill()