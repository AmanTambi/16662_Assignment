import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Lab2'))

import Franka
import numpy as np
import random
import pickle
import RobotUtil as rt
import time

random.seed(13)

mybot = Franka.FrankArm()

# Physical shelf geometry measured from the lab setup.
# These match the simulation scene (Lab2/roadmap_builder.py) which was
# modelled after the real shelves.
EndofTable = 0.55 + 0.135 + 0.05

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
    # right shelf has 3 levels
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

# Pre-compute OBB collision geometry for all obstacles
pointsObs = []
axesObs = []
for _, pos, size in BLOCKS:
    envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0, 0., 0.], pos), size)
    pointsObs.append(envpoints)
    axesObs.append(envaxes)

prmVertices = []
prmEdges = []

# Output file — kept separate from Lab2's myPRM3.p so they don't clobber each other
PRM_FILE = os.path.join(os.path.dirname(__file__), "team12_PRM.p")


def FindKNN(q_new, k, vertices):
    """Return indices of the k nearest nodes by Euclidean distance in joint space."""
    dists = np.linalg.norm(np.array(vertices) - q_new, axis=1)
    return np.argsort(dists)[:k]


def PRMGenerator(n_nodes=1000, k_neighbors=10):
    """
    Build a PRM roadmap for the real-robot shelf workspace.

    Sampling bounds are tightened relative to the full joint limits so that
    random configs land in the half of the workspace in front of the robot
    (x > 0) and below the top shelf (z < 0.95 m) — consistent with what
    we observed from the calibrated grasp poses in poses.txt.
    """
    global prmVertices, prmEdges

    obs_pts = np.array(pointsObs)
    obs_ax  = np.array(axesObs)

    # Conservative joint-space sampling bounds (same as Lab2 — they work well)
    qmin = [-1.57, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    qmax = [ 1.57,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]

    start = time.time()
    while len(prmVertices) < n_nodes:
        q_new = np.random.uniform(low=qmin, high=qmax)

        # Workspace filter: end-effector must be in front of the robot and
        # within the reachable shelf volume.  The top shelf pose in poses.txt
        # has z ≈ 0.58 m; we allow up to 0.95 m to give the planner room to
        # arc over obstacles.
        Tcurr, _ = mybot.ForwardKin(q_new)
        ee_x = Tcurr[-1][0, 3]
        ee_z = Tcurr[-1][2, 3]
        if ee_x < 0.0 or ee_z < 0.0 or ee_z > 0.95:
            continue

        # Skip configurations that collide with any shelf/table obstacle
        if mybot.DetectCollision(q_new, obs_pts, obs_ax):
            continue

        if len(prmVertices) == 0:
            prmVertices.append(q_new)
            prmEdges.append([])
            continue

        # Connect to up to k_neighbors nearest existing nodes if the straight
        # line between them in joint space is collision-free
        knn = FindKNN(q_new, k_neighbors, prmVertices)
        prmVertices.append(q_new)
        prmEdges.append([])
        new_idx = len(prmVertices) - 1

        if new_idx % 100 == 0:
            elapsed = time.time() - start
            print(f"  {new_idx}/{n_nodes} nodes  ({elapsed:.1f}s)")

        for idx in knn:
            q_near = prmVertices[idx]
            if not mybot.DetectCollisionEdge(q_new, q_near, obs_pts, obs_ax):
                prmEdges[new_idx].append(idx)
                prmEdges[idx].append(new_idx)

    print(f"\nBuild complete: {len(prmVertices)} nodes in {time.time()-start:.1f}s")

    with open(PRM_FILE, 'wb') as f:
        pickle.dump(prmVertices, f)
        pickle.dump(prmEdges, f)
        pickle.dump(obs_pts, f)
        pickle.dump(obs_ax, f)

    print(f"Roadmap saved to {PRM_FILE}")


if __name__ == "__main__":
    np.random.seed(13)
    PRMGenerator()
