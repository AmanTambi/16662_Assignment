import numpy as np
import Franka
import RobotUtil as rt
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

BLOCK_NAMES = [block[0] for block in BLOCKS]
ARM_BLOCK_NAMES = [
    "Base Block 0", "Base Block 1", "Base Block 2", "Joint 1 Block",
    "Joint 1 Block 2", "Joint 2 Block", "Joint 2-3 Link", "Joint 3 Block",
    "Joint 3 Block 2", "Joint 4 Block", "Joint 4-5 Link", "Joint 5 Link",
    "Gripper Block 0", "Gripper Block 1"
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

# Initialize robot
print("Initializing FrankArm...")
robot = Franka.FrankArm()

# Get show_object_in_camera joint angles
q_camera = FRANKA_POSES["show_object_in_camera"]["joints"]
print(f"\nTesting joint angles: {q_camera}")

# Build obstacle model
print("\nBuilding obstacle model...")
pointsObs, axesObs = build_obstacle_model(BLOCKS)
print(f"Number of environment obstacles: {len(BLOCKS)}")

# Compute collision block points
print(f"\nComputing collision block points...")
robot.CompCollisionBlockPoints(q_camera)

# Check collision with detailed info
print(f"\nDetailed collision check:")
collision_found = False
for i in range(len(robot.Cpoints)):
    arm_block_name = ARM_BLOCK_NAMES[i] if i < len(ARM_BLOCK_NAMES) else f"Block {i}"
    for j in range(len(pointsObs)):
        env_block_name = BLOCK_NAMES[j]
        if rt.CheckBoxBoxCollision(robot.Cpoints[i], robot.Caxes[i], pointsObs[j], axesObs[j]):
            print(f"  ❌ COLLISION: {arm_block_name} (block {i}) ↔ {env_block_name} (obstacle {j})")
            collision_found = True

if not collision_found:
    print("  ✓ No collisions detected")
else:
    print(f"\n⚠️  Collision detected at show_object_in_camera pose!")
