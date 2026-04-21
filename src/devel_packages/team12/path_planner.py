import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Lab2'))

import numpy as np
import heapq
import pickle
import random
import time

import Franka
import RobotUtil as rt

random.seed(13)

mybot = Franka.FrankArm()

PRM_FILE = os.path.join(os.path.dirname(__file__), "team12_PRM.p")

# Convenience: joint-space home used as a default test config
HOME_CONFIG = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])


def PRMQuery(qInit, qGoal, shortcut_iters=200, k_connect=10):
    """
    Query the team12 PRM with A* and return a shortened waypoint list.

    Parameters
    ----------
    qInit : array-like, shape (7,)
        Starting joint configuration.
    qGoal : array-like, shape (7,)
        Goal joint configuration.
    shortcut_iters : int
        Number of random shortcut attempts after finding the path.
    k_connect : int
        Maximum number of roadmap nodes to try when connecting start/goal.

    Returns
    -------
    list of np.ndarray  –  waypoints [qInit, ..., qGoal]
    None                –  if no path was found
    """
    qInit = np.array(qInit, dtype=float)
    qGoal = np.array(qGoal, dtype=float)

    if not os.path.exists(PRM_FILE):
        raise FileNotFoundError(
            f"PRM file not found: {PRM_FILE}\n"
            "Run roadmap_builder.py first to generate it."
        )

    with open(PRM_FILE, 'rb') as f:
        prmVertices = pickle.load(f)
        prmEdges    = pickle.load(f)
        pointsObs   = pickle.load(f)
        axesObs     = pickle.load(f)

    num_nodes, num_edges, num_components = rt.AnalyzeGraph(prmVertices, prmEdges)
    print(f"PRM: {num_nodes} nodes  {num_edges} edges  {num_components} components")

    # ----- connect qInit to the roadmap -----
    neighInit = _connect_config(qInit, prmVertices, pointsObs, axesObs, k_connect)
    # ----- connect qGoal to the roadmap -----
    neighGoal = _connect_config(qGoal, prmVertices, pointsObs, axesObs, k_connect)

    print(f"Init neighbours: {len(neighInit)},  Goal neighbours: {len(neighGoal)}")

    if not neighInit:
        print("Could not connect start config to roadmap.")
        return None
    if not neighGoal:
        print("Could not connect goal config to roadmap.")
        return None

    # ----- A* on the roadmap -----
    heuristic = [np.linalg.norm(np.array(v) - qGoal) for v in prmVertices]
    g_cost    = [float('inf')] * len(prmVertices)
    parent    = [None] * len(prmVertices)

    open_set = []
    for n in neighInit:
        g = np.linalg.norm(np.array(prmVertices[n]) - qInit)
        g_cost[n] = g
        heapq.heappush(open_set, (g + heuristic[n], n))

    closed_set = set()
    goal_node  = None

    t0 = time.time()
    while open_set:
        _, curr = heapq.heappop(open_set)
        if curr in closed_set:
            continue
        closed_set.add(curr)

        if curr in neighGoal:
            goal_node = curr
            break

        for nbr in prmEdges[curr]:
            if nbr in closed_set:
                continue
            cost = np.linalg.norm(
                np.array(prmVertices[nbr]) - np.array(prmVertices[curr]))
            new_g = g_cost[curr] + cost
            if new_g < g_cost[nbr]:
                g_cost[nbr]  = new_g
                parent[nbr]  = curr
                heapq.heappush(open_set, (new_g + heuristic[nbr], nbr))

    print(f"A* in {time.time()-t0:.4f}s")

    if goal_node is None:
        print("No path found — PRM may be disconnected between start and goal.")
        return None

    # ----- reconstruct path through roadmap -----
    path = [goal_node]
    node = parent[goal_node]
    while node is not None:
        path.insert(0, node)
        node = parent[node]

    waypoints = [qInit] + [prmVertices[i] for i in path] + [qGoal]

    # ----- random shortcutting -----
    for _ in range(shortcut_iters):
        if len(waypoints) <= 2:
            break
        i = random.randint(0, len(waypoints) - 3)
        j = random.randint(i + 2, len(waypoints) - 1)
        if not mybot.DetectCollisionEdge(waypoints[i], waypoints[j], pointsObs, axesObs):
            waypoints = waypoints[:i+1] + waypoints[j:]

    print(f"Plan: {len(waypoints)} waypoints (after shortcutting)")
    return waypoints


def _connect_config(q, prmVertices, pointsObs, axesObs, k):
    """
    Return up to k roadmap node indices that have a collision-free straight
    edge to config q, ordered by distance.
    """
    dists = [(i, np.linalg.norm(np.array(prmVertices[i]) - q))
             for i in range(len(prmVertices))]
    dists.sort(key=lambda x: x[1])

    neighbours = []
    for idx, _ in dists:
        if not mybot.DetectCollisionEdge(prmVertices[idx], q, pointsObs, axesObs):
            neighbours.append(idx)
        if len(neighbours) >= k:
            break
    return neighbours


if __name__ == "__main__":
    # Quick smoke-test: plan from home to a reachable config
    qGoal = np.array([-0.5, -0.8, 0.5, -2.2, 0.3, 1.8, 0.5])
    plan = PRMQuery(HOME_CONFIG, qGoal)
    if plan:
        print("Success — waypoints:")
        for i, wp in enumerate(plan):
            print(f"  [{i}] {np.round(wp, 3)}")
