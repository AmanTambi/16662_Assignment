import numpy as np
import random
import RobotUtil as rt


def build_obstacle_model(blocks):
    #Convert block descriptions [name, center, half_extents] to collision format
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


#RRT planner
def plan_rrt(q_start, q_goal, robot, pointsObs, axesObs,
             max_iter=5000, step_size=0.08, goal_bias=0.08, seed=42):

    random.seed(seed)
    np.random.seed(seed)

    q_start = list(q_start)
    q_goal = list(q_goal)

    vertices = [q_start]
    parents = [0]
    found = False

    for iteration in range(max_iter):
        if random.random() < goal_bias:
            q_rand = list(q_goal)
        else:
            q_rand = robot.SampleRobotConfig()

        nn_idx = rt.FindNearest(vertices, q_rand)
        q_near = vertices[nn_idx]

        q_curr = list(q_near)
        curr_idx = nn_idx

        direction = np.array(q_rand) - np.array(q_curr)
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            continue

        step = min(step_size, dist)
        q_new = (np.array(q_curr) + (direction / dist) * step).tolist()

        for j in range(7):
            q_new[j] = max(robot.qmin[j], min(robot.qmax[j], q_new[j]))

        if robot.DetectCollision(q_new, pointsObs, axesObs):
            continue
        if robot.DetectCollisionEdge(q_curr, q_new, pointsObs, axesObs):
            continue

        vertices.append(q_new)
        parents.append(curr_idx)
        new_idx = len(vertices) - 1

        goal_dist = np.linalg.norm(np.array(q_new) - np.array(q_goal))
        if goal_dist < step_size:
            if not robot.DetectCollision(q_goal, pointsObs, axesObs):
                if not robot.DetectCollisionEdge(q_new, q_goal, pointsObs, axesObs):
                    vertices.append(q_goal)
                    parents.append(new_idx)
                    found = True
                    break

    if not found:
        return None

    #extract path by backtracking parents
    path = []
    idx = len(vertices) - 1
    while True:
        path.insert(0, vertices[idx])
        if idx == 0:
            break
        idx = parents[idx]

    path = shorten_path(path, robot, pointsObs, axesObs, iterations=200)
    return path


def shorten_path(path, robot, pointsObs, axesObs, iterations=200):
    #random shortcutting
    for _ in range(iterations):
        if len(path) <= 2:
            break
        idx_a = random.randint(0, len(path) - 2)
        idx_b = random.randint(idx_a + 1, len(path) - 1)
        if idx_b <= idx_a + 1:
            continue

        q_a = path[idx_a]
        q_b = path[idx_b]

        collision = False
        num_checks = max(2, int(np.linalg.norm(np.array(q_b) - np.array(q_a)) / 0.05))
        for s in np.linspace(0, 1, num_checks + 1):
            q_test = [q_a[k] + s * (q_b[k] - q_a[k]) for k in range(7)]
            if robot.DetectCollision(q_test, pointsObs, axesObs):
                collision = True
                break

        if not collision:
            if not robot.DetectCollisionEdge(q_a, q_b, pointsObs, axesObs):
                path = path[:idx_a + 1] + path[idx_b:]

    return path


def plan_direct(q_start, q_goal, robot, pointsObs, axesObs, num_checks=10):
    #direct linear interpolation, returns None if collision found
    for s in np.linspace(0, 1, num_checks + 1):
        q_test = [q_start[k] + s * (q_goal[k] - q_start[k]) for k in range(7)]
        if robot.DetectCollision(q_test, pointsObs, axesObs):
            return None
    return [list(q_start), list(q_goal)]


#PRM planner (adapted from HW2 PRMGenerator.py + PRMQuery.py)
def build_prm(robot, pointsObs, axesObs, n_vertices=1000, k_neighbors=10,
              connect_radius=2.0, seed=13):
    random.seed(seed)
    np.random.seed(seed)

    vertices = []

    #sample collision-free configs
    while len(vertices) < n_vertices:
        q = robot.SampleRobotConfig()
        if not robot.DetectCollision(q, pointsObs, axesObs):
            vertices.append(q)

    #connect each vertex to k nearest collision-free neighbors
    edges = [[] for _ in range(n_vertices)]
    for i in range(n_vertices):
        dists = [(np.linalg.norm(np.array(vertices[i]) - np.array(vertices[j])), j)
                 for j in range(n_vertices) if j != i]
        dists.sort()
        connected = 0
        for dist, j in dists:
            if connected >= k_neighbors or dist > connect_radius:
                break
            if j not in edges[i]:
                if not robot.DetectCollisionEdge(vertices[i], vertices[j], pointsObs, axesObs):
                    edges[i].append(j)
                    edges[j].append(i)
                    connected += 1

    return {'vertices': vertices, 'edges': edges,
            'pointsObs': pointsObs, 'axesObs': axesObs}


def query_prm(roadmap, q_start, q_goal, robot, pointsObs, axesObs,
              connect_radius=2.0):
    #greedy best-first search on PRM graph (from HW2 PRMQuery pattern)
    vertices = roadmap['vertices']
    edges = roadmap['edges']
    n = len(vertices)

    #find roadmap vertices near start and goal
    neigh_start, neigh_goal = [], []
    for i in range(n):
        d_start = np.linalg.norm(np.array(vertices[i]) - np.array(q_start))
        d_goal = np.linalg.norm(np.array(vertices[i]) - np.array(q_goal))
        if d_start < connect_radius:
            if not robot.DetectCollisionEdge(vertices[i], q_start, pointsObs, axesObs):
                neigh_start.append(i)
        if d_goal < connect_radius:
            if not robot.DetectCollisionEdge(vertices[i], q_goal, pointsObs, axesObs):
                neigh_goal.append(i)

    if not neigh_start or not neigh_goal:
        return None

    heuristic = [np.linalg.norm(np.array(v) - np.array(q_goal)) for v in vertices]
    parent = [None] * n
    active = list(neigh_start)
    visited = set(active)

    found = False
    for _ in range(n * 2):
        if any(g in visited for g in neigh_goal):
            found = True
            break

        best_score = float('inf')
        best_node = None
        best_par = None
        for node in active:
            for neighbor in edges[node]:
                if neighbor not in visited:
                    if heuristic[neighbor] < best_score:
                        best_score = heuristic[neighbor]
                        best_node = neighbor
                        best_par = node

        if best_node is None:
            break

        visited.add(best_node)
        active.append(best_node)
        parent[best_node] = best_par

    if not found:
        return None

    #find which goal neighbor was reached and backtrack
    goal_node = None
    for g in neigh_goal:
        if g in visited:
            goal_node = g
            break

    path_indices = [goal_node]
    node = goal_node
    while parent[node] is not None:
        node = parent[node]
        path_indices.insert(0, node)

    path = [list(q_start)]
    for idx in path_indices:
        path.append(vertices[idx])
    path.append(list(q_goal))

    path = shorten_path(path, robot, pointsObs, axesObs, iterations=200)
    return path


#plan function factories - these return a plan_fn(q_start, q_goal, label) -> path
def make_plan_fn(robot, pointsObs, axesObs, rrt_max_iter=5000):
    #tries direct interpolation first, falls back to RRT
    def plan_fn(q_start, q_goal, label=""):
        path = plan_direct(q_start, q_goal, robot, pointsObs, axesObs)
        if path is not None:
            return path
        path = plan_rrt(q_start, q_goal, robot, pointsObs, axesObs,
                        max_iter=rrt_max_iter)
        if path is not None:
            return path
        return None  # Disabled fallback - return None on planning failure
    return plan_fn


def make_prm_plan_fn(roadmap, robot, pointsObs, axesObs):
    #tries direct first, then PRM query, then RRT fallback
    def plan_fn(q_start, q_goal, label=""):
        path = plan_direct(q_start, q_goal, robot, pointsObs, axesObs)
        if path is not None:
            return path
        path = query_prm(roadmap, q_start, q_goal, robot, pointsObs, axesObs)
        if path is not None:
            return path
        path = plan_rrt(q_start, q_goal, robot, pointsObs, axesObs)
        if path is not None:
            return path
        return [list(q_start)]
    return plan_fn
