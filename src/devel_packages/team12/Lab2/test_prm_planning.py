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

HOME_Q = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]

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
    """Check if any configuration in path collides"""
    for i, q in enumerate(path):
        if robot.DetectCollision(q, pointsObs, axesObs):
            print(f"  ❌ Collision found at waypoint {i}: {q}")
            return True
    return False

# Initialize
print("="*60)
print("PRM Planning Test: HOME to show_object_in_camera")
print("="*60)

print("\n1. Initializing robot...")
robot = Franka.FrankArm()

print("\n2. Building obstacle model...")
pointsObs, axesObs = build_obstacle_model(BLOCKS)
print(f"   - Number of obstacles: {len(BLOCKS)}")

print("\n3. Building PRM roadmap (n_vertices=500, k_neighbors=10)...")
print("   This may take a moment...")
start_time = time.time()
roadmap = build_prm(robot, pointsObs, axesObs, n_vertices=500, k_neighbors=10)
build_time = time.time() - start_time
print(f"   ✓ PRM built in {build_time:.2f}s")
print(f"   - Roadmap vertices: {len(roadmap['vertices'])}")

print("\n4. Creating plan function...")
plan_fn = make_prm_plan_fn(roadmap, robot, pointsObs, axesObs)

print("\n5. Planning from HOME to show_object_in_camera...")
q_start = HOME_Q
q_goal = FRANKA_POSES["show_object_in_camera"]["joints"].tolist()

print(f"   Start (HOME): {q_start}")
print(f"   Goal (CAMERA): {q_goal}")

start_time = time.time()
path = plan_fn(q_start, q_goal, label="home_to_camera")
plan_time = time.time() - start_time

if path is None or len(path) == 0:
    print(f"\n❌ Planning FAILED - No path found!")
    sys.exit(1)

print(f"\n✓ Planning succeeded in {plan_time:.2f}s")
print(f"   - Path length: {len(path)} waypoints")

print("\n6. Verifying path for collisions...")
path_collision = check_path_collision(path, robot, pointsObs, axesObs)

if path_collision:
    print(f"\n⚠️  WARNING: Path contains collisions!")
    sys.exit(1)
else:
    print(f"   ✓ Path is collision-free")

print("\n7. Verifying start and goal configurations...")
start_collision = robot.DetectCollision(q_start, pointsObs, axesObs)
goal_collision = robot.DetectCollision(q_goal, pointsObs, axesObs)

print(f"   Start config collision: {'❌ YES' if start_collision else '✓ NO'}")
print(f"   Goal config collision: {'❌ YES' if goal_collision else '✓ NO'}")

if goal_collision:
    print(f"\n⚠️  Goal configuration has collision!")
    print(f"    The camera pose may need adjustment.")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print(f"Planning result: ✓ SUCCESS")
print(f"Path length: {len(path)} waypoints")
print(f"Path is collision-free: ✓ YES")
if goal_collision:
    print(f"Note: Goal pose has collision - arm hits shelf")
else:
    print(f"Note: Goal pose is collision-free")
print("="*60)
