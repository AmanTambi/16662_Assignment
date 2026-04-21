import numpy as np
import time
import Franka
import RobotUtil as rt

# Try to import FrankaArm from frankapy (real robot interface)
try:
    from frankapy import FrankaArm as FrankaArmReal
    REAL_ROBOT_AVAILABLE = True
except ImportError:
    REAL_ROBOT_AVAILABLE = False

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

def check_pose_collision_detailed(q, robot, pointsObs, axesObs):
    """Check collision and return detailed collision info"""
    robot.CompCollisionBlockPoints(q)
    collisions = []
    
    for i in range(len(robot.Cpoints)):
        arm_block_name = ARM_BLOCK_NAMES[i] if i < len(ARM_BLOCK_NAMES) else f"Block {i}"
        for j in range(len(pointsObs)):
            env_block_name = BLOCK_NAMES[j]
            if rt.CheckBoxBoxCollision(robot.Cpoints[i], robot.Caxes[i], pointsObs[j], axesObs[j]):
                collisions.append((i, j, arm_block_name, env_block_name))
    
    return collisions

def print_collision_status(q, collisions, is_collision, iteration):
    """Pretty print collision information"""
    status = "❌ COLLISION" if is_collision else "✓ CLEAR"
    
    # Format joint angles nicely
    q_str = ", ".join([f"{angle:7.4f}" for angle in q])
    
    if is_collision:
        collision_str = f"({len(collisions)} collision(s))"
        print(f"[{iteration:4d}] {status:15} {collision_str:20} | [{q_str}]")
        for arm_idx, env_idx, arm_name, env_name in collisions:
            print(f"         → {arm_name:20} ↔ {env_name:20}")
    else:
        print(f"[{iteration:4d}] {status:15} {'':20} | [{q_str}]")

def main():
    print("="*100)
    print("REAL-TIME JOINT COLLISION CHECKER - Reading from FrankaArm")
    print("="*100)
    
    print("\nInitializing...")
    
    # Initialize simulation robot
    robot = Franka.FrankArm()
    pointsObs, axesObs = build_obstacle_model(BLOCKS)
    print(f"✓ Simulation robot initialized with {len(robot.Cdesc)} collision blocks")
    print(f"✓ Environment has {len(BLOCKS)} obstacles")
    
    # Try to connect to real robot
    if REAL_ROBOT_AVAILABLE:
        print("\nAttempting to connect to real FrankaArm...")
        try:
            fa = FrankaArmReal()
            print("✓ Connected to real FrankaArm!")
            use_real_robot = True
        except Exception as e:
            print(f"⚠️  Could not connect to real robot: {e}")
            print("   Will read poses from stdin instead")
            use_real_robot = False
    else:
        print("\n⚠️  frankapy not available")
        print("   Will read poses from stdin")
        use_real_robot = False
    
    print("\n" + "="*100)
    if use_real_robot:
        print("MONITORING REAL ROBOT JOINT POSES")
        print("Press Ctrl+C to stop")
    else:
        print("READING JOINT POSES FROM STDIN")
        print("Enter 7 space-separated joint angles per line or 'exit' to quit")
    print("="*100 + "\n")
    
    iteration = 0
    last_collision_state = None
    
    try:
        while True:
            iteration += 1
            
            if use_real_robot:
                # Get current joint configuration from real robot
                try:
                    q = list(fa.get_joints())
                except Exception as e:
                    print(f"Error reading robot joints: {e}")
                    continue
            else:
                # For fallback, read from user input
                try:
                    user_input = input(f"[{iteration}] Enter 7 joint angles: ").strip()
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        break
                    
                    values = user_input.split()
                    if len(values) != 7:
                        print(f"Error: Expected 7 values, got {len(values)}")
                        iteration -= 1
                        continue
                    q = [float(v) for v in values]
                except ValueError as e:
                    print(f"Error parsing input: {e}")
                    iteration -= 1
                    continue
                except EOFError:
                    break
            
            # Check collision
            collisions = check_pose_collision_detailed(q, robot, pointsObs, axesObs)
            is_collision = len(collisions) > 0
            
            # Print status
            print_collision_status(q, collisions, is_collision, iteration)
            
            # Alert on collision state change
            if last_collision_state is not None and last_collision_state != is_collision:
                if is_collision:
                    print("         ⚠️  WARNING: Entered collision state!")
                else:
                    print("         ✓ Cleared collision")
            
            last_collision_state = is_collision
            
            # Small delay if using real robot (to avoid hammering the controller)
            if use_real_robot:
                time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*100)
        print("Monitoring stopped by user")
        print("="*100)
    
    if use_real_robot:
        try:
            fa.stop_skill()
        except:
            pass

if __name__ == "__main__":
    main()
