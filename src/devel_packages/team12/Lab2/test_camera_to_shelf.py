import numpy as np
import sys
import time
import Franka
import RobotUtil as rt
from planner import build_prm, make_prm_plan_fn
from poses import FRANKA_POSES

# BLOCKS definition from main.py
EndofTable = 0.55 + 0.135 + 0.05
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

def build_obstacle_model(blocks):
    """Convert block descriptions to collision format"""
    pointsObs, axesObs = [], []
    for block in blocks:
        center = block[1]
        half_ext = block[2]
        H = rt.rpyxyz2H([0, 0, 0], center)
        Dim = [2 * half_ext[0], 2 * half_ext[1], 2 * half_ext[2]]
        pts, axes = rt.BlockDesc2Points(H, Dim)
        pointsObs.append(pts)
        axesObs.append(axes)
    return pointsObs, axesObs

def check_path_collision(path, robot, pointsObs, axesObs):
    """Check if any configuration in path collides and return details"""
    collisions = []
    for i, q in enumerate(path):
        if robot.DetectCollision(q, pointsObs, axesObs):
            collisions.append(i)
    return collisions

# Initialize
print("="*75)
print("PRM PLANNING TEST: show_object_in_camera → top_shelf_middle")
print("="*75)

print("\n1. Initializing robot...")
robot = Franka.FrankArm()

print("\n2. Building obstacle model...")
pointsObs, axesObs = build_obstacle_model(BLOCKS)
print(f"   ✓ {len(BLOCKS)} obstacles loaded")

print("\n3. Building PRM roadmap (n_vertices=300, k_neighbors=10)...")
start_time = time.time()
roadmap = build_prm(robot, pointsObs, axesObs, n_vertices=300, k_neighbors=10)
build_time = time.time() - start_time
print(f"   ✓ Built in {build_time:.2f}s with {len(roadmap['vertices'])} vertices")

print("\n4. Creating PRM plan function...")
plan_fn = make_prm_plan_fn(roadmap, robot, pointsObs, axesObs)

# Get poses
q_camera = FRANKA_POSES["show_object_in_camera"]["joints"].tolist()
q_shelf = FRANKA_POSES["top_shelf_middle"]["joints"].tolist()

print("\n" + "="*75)
print("PLANNING TEST: show_object_in_camera → top_shelf_middle")
print("="*75)

print(f"\nStart (CAMERA): {q_camera}")
print(f"Goal (SHELF):   {q_shelf}")

# Check collision status
camera_collision = robot.DetectCollision(q_camera, pointsObs, axesObs)
shelf_collision = robot.DetectCollision(q_shelf, pointsObs, axesObs)

print(f"\nConfiguration status:")
print(f"  Camera pose collision: {'❌ YES' if camera_collision else '✓ NO'}")
print(f"  Shelf pose collision:  {'❌ YES' if shelf_collision else '✓ NO'}")

# Plan path
print(f"\nPlanning path (this may take 30-90 seconds)...")
start_time = time.time()
path = plan_fn(q_camera, q_shelf, label="camera_to_shelf")
plan_time = time.time() - start_time

print(f"✓ Planning completed in {plan_time:.2f}s")
print(f"  Path has {len(path)} waypoints")

if path and len(path) > 1:
    # Check for collisions in path
    print(f"\nVerifying path collision-free status...")
    collisions = check_path_collision(path, robot, pointsObs, axesObs)
    
    if collisions:
        print(f"  ❌ Path contains collisions at {len(collisions)} waypoint(s): {collisions[:10]}")
        if len(collisions) > 10:
            print(f"     (and {len(collisions) - 10} more)")
        
        # Show first collision details
        first_collision_idx = collisions[0]
        print(f"\n  First collision at waypoint {first_collision_idx}:")
        print(f"    Joint angles: {path[first_collision_idx]}")
    else:
        print(f"  ✓ Entire path is collision-free!")
    
    # Show path structure
    print(f"\nPath structure:")
    print(f"  - Total waypoints: {len(path)}")
    print(f"  - Start: {path[0]}")
    print(f"  - Goal:  {path[-1]}")
    if len(path) > 10:
        print(f"  - (showing first/last 3 of {len(path)} waypoints)")
        for i in range(min(3, len(path))):
            print(f"    [{i}]: {path[i]}")
        print(f"    ...")
        for i in range(max(3, len(path)-3), len(path)):
            print(f"    [{i}]: {path[i]}")
else:
    print(f"  ⚠️  Degenerate path with {len(path)} waypoints (fallback path)")

print("\n" + "="*75)
print("SUMMARY")
print("="*75)
print(f"Planning: ✓ COMPLETED")
print(f"Path length: {len(path)} waypoints")
if path and len(path) > 1:
    collisions = check_path_collision(path, robot, pointsObs, axesObs)
    if collisions:
        print(f"Path safety: ❌ Contains {len(collisions)} collision(s)")
    else:
        print(f"Path safety: ✓ COLLISION-FREE")
        print(f"\n✓ Successfully planned collision-free path from CAMERA to SHELF!")
else:
    print(f"Path safety: ⚠️  Degenerate path - goal may be unreachable")
print("="*75)
