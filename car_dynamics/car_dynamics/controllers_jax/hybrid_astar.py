"""
Hybrid A* global planner for non-holonomic robots.

Searches in (x, y, theta) space using actual dynamics to generate successors.
Every edge is a dynamically feasible arc, so the resulting path can be tracked
directly by MPPI without heading discontinuities or infeasible turns.

Usage:
    from car_dynamics.controllers_jax import hybrid_astar_plan

    path = hybrid_astar_plan(start, goal, dynamics, obstacle_map, dt=0.15)
"""

import heapq
import numpy as np
import numba


# ---------------------------------------------------------------------------
# Numba-compiled inner loops (unicycle dynamics + collision check)
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _nb_unicycle_expand(state, steer_val, v_default, dt, wheelbase,
                         extend_steps, sdf_grid, sdf_origin_x, sdf_origin_y,
                         inv_res, grid_w, grid_h, collision_margin):
    """Expand one edge: simulate unicycle for extend_steps. Returns (next_state, arc_len, ok)."""
    x, y, theta, v = state[0], state[1], state[2], state[3]
    accel = min(max((v_default - v) * 2.0, -1.0), 1.0)
    arc_length = 0.0

    for _ in range(extend_steps):
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + v * np.tan(steer_val) / wheelbase * dt
        v_new = v + accel * dt

        ix = int((x_new - sdf_origin_x) * inv_res)
        iy = int((y_new - sdf_origin_y) * inv_res)
        if ix < 0 or ix >= grid_w or iy < 0 or iy >= grid_h:
            return np.array([x, y, theta, v], dtype=np.float32), 0.0, False
        if sdf_grid[ix, iy] < collision_margin:
            return np.array([x, y, theta, v], dtype=np.float32), 0.0, False

        dx = x_new - x
        dy = y_new - y
        arc_length += np.sqrt(dx * dx + dy * dy)
        x, y, theta, v = x_new, y_new, theta_new, v_new

    return np.array([x, y, theta, v], dtype=np.float32), arc_length, True


@numba.njit(cache=True)
def _nb_unicycle_goal_shot(state, goal_x, goal_y, goal_theta,
                            v_default, dt, wheelbase,
                            sdf_grid, sdf_origin_x, sdf_origin_y,
                            inv_res, grid_w, grid_h, collision_margin,
                            goal_pos_thr, goal_heading_thr, max_steps):
    """Drive from state to goal with proportional controller. Returns (path_array, n_points, ok)."""
    # Pre-allocate output array
    path = np.zeros((max_steps + 1, 4), dtype=np.float32)
    x, y, theta, v = state[0], state[1], state[2], state[3]
    n = 0

    for _ in range(max_steps):
        dx = goal_x - x
        dy = goal_y - y
        dist = np.sqrt(dx * dx + dy * dy)
        target_heading = np.arctan2(dy, dx)
        heading_err = np.arctan2(np.sin(target_heading - theta), np.cos(target_heading - theta))

        if dist < goal_pos_thr * 3.0:
            final_h_err = np.arctan2(np.sin(goal_theta - theta), np.cos(goal_theta - theta))
            heading_err = 0.5 * heading_err + 0.5 * final_h_err

        steer_cmd = min(max(heading_err * 3.0, -1.0), 1.0)
        target_v = min(v_default, dist * 2.0)
        accel = min(max((target_v - v) * 2.0, -1.0), 1.0)

        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + v * np.tan(steer_cmd) / wheelbase * dt
        v_new = v + accel * dt

        ix = int((x_new - sdf_origin_x) * inv_res)
        iy = int((y_new - sdf_origin_y) * inv_res)
        if ix < 0 or ix >= grid_w or iy < 0 or iy >= grid_h:
            return path, 0, False
        if sdf_grid[ix, iy] < collision_margin:
            return path, 0, False

        x, y, theta, v = x_new, y_new, theta_new, v_new
        path[n, 0] = x
        path[n, 1] = y
        path[n, 2] = theta
        path[n, 3] = v
        n += 1

        p_err = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        h_err = abs(np.arctan2(np.sin(theta - goal_theta), np.cos(theta - goal_theta)))
        if p_err < goal_pos_thr * 0.5 and h_err < goal_heading_thr * 0.5:
            # Add exact goal
            path[n, 0] = goal_x
            path[n, 1] = goal_y
            path[n, 2] = goal_theta
            path[n, 3] = 0.0
            n += 1
            return path, n, True

    return path, 0, False


@numba.njit(cache=True)
def _nb_holonomic_expand(state, steer_val, v_default, dt,
                          extend_steps, sdf_grid, sdf_origin_x, sdf_origin_y,
                          inv_res, grid_w, grid_h, collision_margin):
    """Expand for holonomic/diff_drive (3D state)."""
    x, y, theta = state[0], state[1], state[2]
    arc_length = 0.0
    c = np.cos(theta)
    s = np.sin(theta)

    for _ in range(extend_steps):
        x_new = x + (v_default * c) * dt
        y_new = y + (v_default * s) * dt
        theta_new = theta + steer_val * dt

        ix = int((x_new - sdf_origin_x) * inv_res)
        iy = int((y_new - sdf_origin_y) * inv_res)
        if ix < 0 or ix >= grid_w or iy < 0 or iy >= grid_h:
            return np.array([x, y, theta], dtype=np.float32), 0.0, False
        if sdf_grid[ix, iy] < collision_margin:
            return np.array([x, y, theta], dtype=np.float32), 0.0, False

        ddx = x_new - x
        ddy = y_new - y
        arc_length += np.sqrt(ddx * ddx + ddy * ddy)
        x, y, theta = x_new, y_new, theta_new
        c = np.cos(theta)
        s = np.sin(theta)

    return np.array([x, y, theta], dtype=np.float32), arc_length, True


def hybrid_astar_plan(
    start,
    goal,
    step_fn,
    sdf_grid,
    sdf_origin,
    sdf_resolution,
    dt=0.15,
    v_default=0.5,
    steer_set=None,
    extend_steps=5,
    grid_xy_res=0.1,
    grid_theta_res=None,
    collision_margin=0.05,
    goal_pos_threshold=0.2,
    goal_heading_threshold=0.5,
    max_iterations=100000,
    dynamics_kwargs=None,
):
    """
    Plan a dynamically feasible path using Hybrid A*.

    Args:
        start: (D,) start state (at least [x, y, theta]).
        goal: (D,) goal state.
        step_fn: callable(state, action) -> next_state. Numpy single-step dynamics.
        sdf_grid: (W, H) numpy SDF array.
        sdf_origin: (2,) world coords of grid[0, 0].
        sdf_resolution: meters per cell.
        dt: timestep for dynamics (baked into step_fn for some models).
        v_default: forward velocity for expansion (used to construct actions).
        steer_set: list of steering values to try. Default: 7 values in [-1, 1].
        extend_steps: dynamics steps per edge (longer = fewer nodes, coarser).
        grid_xy_res: spatial resolution for state deduplication (meters).
        grid_theta_res: heading resolution for deduplication (radians). Default: 2*pi/72.
        collision_margin: minimum SDF distance for collision-free.
        goal_pos_threshold: position tolerance for reaching goal (meters).
        goal_heading_threshold: heading tolerance (radians).
        max_iterations: maximum nodes to expand.

    Returns:
        List of (D,) numpy states from start to goal, or None if no path found.
    """
    start = np.asarray(start, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    sdf_grid = np.asarray(sdf_grid, dtype=np.float32)
    sdf_origin = np.asarray(sdf_origin, dtype=np.float32)[:2]
    D = len(start)

    if steer_set is None:
        steer_set = [-0.8, -0.3, 0.0, 0.3, 0.8]

    if grid_theta_res is None:
        grid_theta_res = 2 * np.pi / 72  # 5-degree bins

    goal_pos = goal[:2]
    goal_theta = goal[2] if D > 2 else 0.0
    grid_w, grid_h = sdf_grid.shape

    # Build action set: for unicycle [accel, steer], for holonomic [vx, vy, omega], etc.
    # We detect action format from step_fn's expected input.
    # For generality, we construct actions based on state dims:
    #   D=3 (holonomic): action = [v_fwd, v_side, omega] — use steer as omega
    #   D=4 (unicycle):  action = [accel, steer]
    #   D=3 (diff_drive): action = [vl, vr]
    # The caller should pick the right step_fn. We build actions for unicycle by default.
    state_dims = D

    def _make_action(steer_val):
        """Build an action vector for the given steering value."""
        if state_dims == 4:
            # Unicycle: [accel, steer]. Use small accel to maintain v_default.
            return np.array([0.0, steer_val], dtype=np.float32)
        elif state_dims == 3:
            # Holonomic or diff_drive: [v_fwd, 0, omega]
            return np.array([v_default, 0.0, steer_val], dtype=np.float32)
        else:
            return np.array([v_default, steer_val], dtype=np.float32)

    # Forward-only actions (reverse doubles branching for little benefit)
    action_set = [(v_default, s) for s in steer_set]

    def _discretize(state):
        """Map continuous state to discrete grid cell for deduplication."""
        ix = int(round((state[0] - sdf_origin[0]) / grid_xy_res))
        iy = int(round((state[1] - sdf_origin[1]) / grid_xy_res))
        theta = state[2] if len(state) > 2 else 0.0
        itheta = int(round(theta / grid_theta_res)) % int(round(2 * np.pi / grid_theta_res))
        return (ix, iy, itheta)

    def _heuristic(state):
        """Admissible heuristic: Euclidean distance to goal (ignores obstacles)."""
        return float(np.linalg.norm(state[:2] - goal_pos))

    def _collision_check(state):
        """Check if state is collision-free in SDF."""
        ix = int((state[0] - sdf_origin[0]) / sdf_resolution)
        iy = int((state[1] - sdf_origin[1]) / sdf_resolution)
        if ix < 0 or ix >= grid_w or iy < 0 or iy >= grid_h:
            return False
        return sdf_grid[ix, iy] >= collision_margin

    def _is_goal(state):
        """Check if state is within goal tolerance."""
        pos_err = np.linalg.norm(state[:2] - goal_pos)
        if pos_err > goal_pos_threshold:
            return False
        if D > 2:
            h_err = abs(np.arctan2(np.sin(state[2] - goal_theta),
                                    np.cos(state[2] - goal_theta)))
            if h_err > goal_heading_threshold:
                return False
        return True

    inv_res = 1.0 / sdf_resolution
    sdf_ox, sdf_oy = float(sdf_origin[0]), float(sdf_origin[1])
    wheelbase = float(dynamics_kwargs.get('wheelbase', 0.3)) if dynamics_kwargs else 0.3

    def _expand(state, v_cmd, steer_val):
        """Simulate dynamics for extend_steps using Numba-compiled inner loop."""
        if state_dims == 4:
            s_new, arc, ok = _nb_unicycle_expand(
                state, steer_val, v_cmd, dt, wheelbase,
                extend_steps, sdf_grid, sdf_ox, sdf_oy,
                inv_res, grid_w, grid_h, collision_margin)
        else:
            s_new, arc, ok = _nb_holonomic_expand(
                state, steer_val, v_default, dt,
                extend_steps, sdf_grid, sdf_ox, sdf_oy,
                inv_res, grid_w, grid_h, collision_margin)
        if not ok:
            return None, 0.0
        return s_new, arc

    def _goal_shot(state):
        """Try to drive from state to exact goal using Numba-compiled controller."""
        if state_dims == 4:
            max_shot = min(int(np.linalg.norm(goal_pos - state[:2]) / (v_default * dt * 0.5)) + 20, 200)
            path_arr, n, ok = _nb_unicycle_goal_shot(
                state, goal_pos[0], goal_pos[1], goal_theta,
                v_default, dt, wheelbase,
                sdf_grid, sdf_ox, sdf_oy,
                inv_res, grid_w, grid_h, collision_margin,
                goal_pos_threshold, goal_heading_threshold, max_shot)
            if not ok:
                return None
            return [path_arr[i].copy() for i in range(n)]
        else:
            # Fallback for non-unicycle: simple proportional controller in Python
            s = state.copy()
            shot_path = []
            max_shot = min(int(np.linalg.norm(goal_pos - s[:2]) / (v_default * dt * 0.5)) + 20, 200)
            for _ in range(max_shot):
                dx, dy = goal_pos[0] - s[0], goal_pos[1] - s[1]
                dist = np.sqrt(dx*dx + dy*dy)
                heading_err = np.arctan2(np.sin(np.arctan2(dy,dx) - s[2]), np.cos(np.arctan2(dy,dx) - s[2]))
                if dist < goal_pos_threshold * 3:
                    fh = np.arctan2(np.sin(goal_theta-s[2]), np.cos(goal_theta-s[2]))
                    heading_err = 0.5*heading_err + 0.5*fh
                speed = min(v_default, dist*2.0)
                steer = np.clip(heading_err*3.0, -1.0, 1.0)
                s_new = step_fn(s, np.array([speed, 0.0, steer], dtype=np.float32))
                ix = int((s_new[0]-sdf_origin[0])*inv_res)
                iy = int((s_new[1]-sdf_origin[1])*inv_res)
                if ix<0 or ix>=grid_w or iy<0 or iy>=grid_h: return None
                if sdf_grid[ix,iy]<collision_margin: return None
                shot_path.append(s_new.copy()); s = s_new
                if np.linalg.norm(s[:2]-goal_pos)<goal_pos_threshold*0.5:
                    shot_path.append(goal[:D].copy()); return shot_path
            return None

    # --- A* search ---
    counter = 0
    start_key = _discretize(start)
    g_cost = {start_key: 0.0}
    parent = {start_key: None}
    state_of = {start_key: start.copy()}

    h0 = _heuristic(start)
    open_set = [(h0, counter, start_key)]
    counter += 1

    while open_set and counter < max_iterations:
        f, _, current_key = heapq.heappop(open_set)

        current_state = state_of[current_key]
        current_g = g_cost[current_key]

        # Analytic expansion: try to drive directly to exact goal
        dist_to_goal = np.linalg.norm(current_state[:2] - goal_pos)
        if dist_to_goal < goal_pos_threshold * 20:  # attempt when within reasonable range
            shot = _goal_shot(current_state)
            if shot is not None:
                # Reconstruct path from start to current node
                path = []
                key = current_key
                while key is not None:
                    path.append(state_of[key])
                    key = parent[key]
                path.reverse()
                # Append the goal-shot segment (all dynamically feasible)
                path.extend(shot)
                return path

        # Expand neighbors
        for v_cmd, steer_val in action_set:
            next_state, arc_len = _expand(current_state, v_cmd, steer_val)
            if next_state is None:
                continue  # collision

            next_key = _discretize(next_state)

            tentative_g = current_g + arc_len
            if next_key in g_cost and tentative_g >= g_cost[next_key]:
                continue  # already found a better path to this cell

            g_cost[next_key] = tentative_g
            parent[next_key] = current_key
            state_of[next_key] = next_state.copy()

            f_cost = tentative_g + _heuristic(next_state)
            heapq.heappush(open_set, (f_cost, counter, next_key))
            counter += 1

    return None  # no path found


def hybrid_astar_global_path(
    start,
    goal,
    dynamics,
    obstacle_map,
    dt=0.15,
    v_default=0.5,
    steer_set=None,
    extend_steps=5,
    grid_xy_res=0.1,
    goal_pos_threshold=0.2,
    goal_heading_threshold=0.5,
    collision_margin=0.05,
    **dynamics_kwargs,
):
    """
    Convenience function matching plan_global_path() interface.

    Args:
        start: (D,) start state.
        goal: (D,) goal state.
        dynamics: 'holonomic', 'unicycle', 'diff_drive', 'robocasa_holonomic',
            or a callable step_fn.
        obstacle_map: ObstacleMap instance.
        dt: timestep.
        v_default: nominal velocity for expansion.
        steer_set: discrete steering inputs to try.
        extend_steps: dynamics steps per graph edge.
        grid_xy_res: grid resolution for deduplication (meters).
        goal_pos_threshold: position tolerance.
        goal_heading_threshold: heading tolerance.
        collision_margin: SDF collision threshold.
        **dynamics_kwargs: passed to dynamics factory.

    Returns:
        List of (D,) numpy states from start to goal, or None.
    """
    from .global_planner import (
        GLOBAL_DYNAMICS_REGISTRY,
        compute_velocity_profile,
    )

    if isinstance(dynamics, str):
        if dynamics not in GLOBAL_DYNAMICS_REGISTRY:
            raise ValueError(f"Unknown dynamics '{dynamics}'. "
                             f"Available: {list(GLOBAL_DYNAMICS_REGISTRY.keys())}")
        entry = GLOBAL_DYNAMICS_REGISTRY[dynamics]
        step_fn = entry['factory'](dt, **dynamics_kwargs)
    elif callable(dynamics):
        step_fn = dynamics
    else:
        raise TypeError(f"dynamics must be str or callable, got {type(dynamics)}")

    sdf_np = np.asarray(obstacle_map.sdf, dtype=np.float32)
    origin_np = np.asarray(obstacle_map.origin, dtype=np.float32)

    path = hybrid_astar_plan(
        start=start,
        goal=goal,
        step_fn=step_fn,
        sdf_grid=sdf_np,
        sdf_origin=origin_np,
        sdf_resolution=float(obstacle_map.resolution),
        dt=dt,
        v_default=v_default,
        steer_set=steer_set,
        extend_steps=extend_steps,
        grid_xy_res=grid_xy_res,
        collision_margin=collision_margin,
        goal_pos_threshold=goal_pos_threshold,
        goal_heading_threshold=goal_heading_threshold,
        dynamics_kwargs=dynamics_kwargs,
    )

    if path is None:
        return None

    # Compute and attach velocity profile
    v_profile = compute_velocity_profile(path)
    for i in range(len(path)):
        wp = path[i]
        if len(wp) < 4:
            path[i] = np.concatenate([wp, [v_profile[i]]])
        else:
            path[i] = wp.copy()
            path[i][3] = v_profile[i]

    return path
