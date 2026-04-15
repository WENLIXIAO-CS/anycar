"""
Kinodynamic RRT global planner for long-range navigation.

Grows a tree of dynamically feasible states using the robot's dynamics model,
then extracts a collision-free path of intermediate waypoints for MPPI to track.

Uses the same dynamics models as MPPI (holonomic, unicycle, diff_drive) so the
global path is feasible by construction -- no geometric-to-kinodynamic gap.

Architecture:
    KinodynamicRRT  -->  coarse dynamically feasible path
           |
    track_path_goal_list  -->  sliding-window goal_list (H+1, D)
           |
    CollisionMPPI  -->  smooth, reactive local tracking

Usage:
    from car_dynamics.controllers_jax import plan_global_path, track_path_goal_list

    path = plan_global_path(start, goal, dynamics='holonomic', obstacle_map=obs_map)
    goal_list = track_path_goal_list(path, current_state, horizon=73)
"""

import numpy as np


# ------------------------------------------------------------------ #
#  Kinodynamic RRT                                                    #
# ------------------------------------------------------------------ #

class KinodynamicRRT:
    """
    Kinodynamic RRT planner on a 2D SDF grid.

    Grows a tree from the start state by sampling random actions, propagating
    through a dynamics model, and checking collisions against the SDF.
    When a node reaches the goal region, the path is extracted and returned.

    The dynamics step_fn must be a pure-numpy callable (not JAX) for speed
    in the inner loop. Use the make_*_step() factories below, or wrap your own.
    """

    def __init__(
        self,
        step_fn,
        state_dims,
        action_dims,
        action_low,
        action_high,
        pos_indices=(0, 1),
        heading_index=2,
    ):
        """
        Args:
            step_fn: callable(state, action) -> next_state.
                Single-step dynamics. state and action are 1D numpy arrays.
            state_dims: dimensionality of state vector.
            action_dims: dimensionality of action vector.
            action_low: (action_dims,) min action values.
            action_high: (action_dims,) max action values.
            pos_indices: which state indices are (x, y) position.
            heading_index: which state index is heading (for wraparound), or None.
        """
        self.step_fn = step_fn
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.pos_idx = list(pos_indices)
        self.heading_index = heading_index

    def plan(
        self,
        start,
        goal,
        sdf_grid,
        sdf_origin,
        sdf_resolution,
        max_iters=5000,
        extend_steps=5,
        n_action_samples=5,
        goal_bias=0.3,
        goal_pos_threshold=0.2,
        goal_heading_threshold=0.5,
        collision_margin=0.05,
        seed=42,
    ):
        """
        Plan a kinodynamically feasible path from start to goal.

        Args:
            start: (D,) start state.
            goal: (D,) goal state.
            sdf_grid: (W, H) numpy SDF array. Positive = free, negative = obstacle.
            sdf_origin: (2,) world coords of grid[0, 0].
            sdf_resolution: meters per cell.
            max_iters: maximum tree expansion iterations.
            extend_steps: dynamics steps per tree extension.
            n_action_samples: actions to try per extension (pick best).
            goal_bias: probability of sampling goal as the random target.
            goal_pos_threshold: position tolerance for reaching goal (meters).
            goal_heading_threshold: heading tolerance for reaching goal (radians).
            collision_margin: minimum SDF value to be considered collision-free.
            seed: random seed.

        Returns:
            List of (D,) numpy states from start to goal, or None if no path found.
        """
        start = np.asarray(start, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        sdf_grid = np.asarray(sdf_grid, dtype=np.float32)
        sdf_origin = np.asarray(sdf_origin, dtype=np.float32)[:2]
        rng = np.random.default_rng(seed)

        # Pre-allocate tree storage
        max_nodes = max_iters + 1
        node_states = np.zeros((max_nodes, self.state_dims), dtype=np.float32)
        node_positions = np.zeros((max_nodes, 2), dtype=np.float32)
        parent_indices = np.full(max_nodes, -1, dtype=np.int32)

        node_states[0] = start
        node_positions[0] = start[self.pos_idx]
        n_nodes = 1

        goal_pos = goal[self.pos_idx]
        grid_w, grid_h = sdf_grid.shape

        # State bounds for random sampling (derived from SDF grid)
        state_lo = np.full(self.state_dims, -np.pi, dtype=np.float32)
        state_hi = np.full(self.state_dims, np.pi, dtype=np.float32)
        state_lo[self.pos_idx] = sdf_origin
        state_hi[self.pos_idx] = sdf_origin + np.array(
            [grid_w, grid_h], dtype=np.float32
        ) * sdf_resolution

        for _ in range(max_iters):
            # --- Sample target (goal-biased) ---
            if rng.random() < goal_bias:
                x_target_pos = goal_pos
            else:
                x_target_pos = rng.uniform(state_lo[self.pos_idx], state_hi[self.pos_idx])

            # --- Nearest neighbor (brute-force on positions, fast for <50K nodes) ---
            dists = np.sum(
                (node_positions[:n_nodes] - x_target_pos) ** 2, axis=1
            )
            nearest_idx = int(np.argmin(dists))
            x_near = node_states[nearest_idx]

            # --- Try multiple actions, pick best ---
            best_state = None
            best_dist = np.inf

            for _ in range(n_action_samples):
                action = rng.uniform(self.action_low, self.action_high)
                state = x_near.copy()
                collision = False

                for _ in range(extend_steps):
                    state = self.step_fn(state, action)
                    pos = state[self.pos_idx]
                    ix = int((pos[0] - sdf_origin[0]) / sdf_resolution)
                    iy = int((pos[1] - sdf_origin[1]) / sdf_resolution)
                    if ix < 0 or ix >= grid_w or iy < 0 or iy >= grid_h:
                        collision = True
                        break
                    if sdf_grid[ix, iy] < collision_margin:
                        collision = True
                        break

                if collision:
                    continue

                ep = state[self.pos_idx]
                d = (ep[0] - x_target_pos[0]) ** 2 + (ep[1] - x_target_pos[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_state = state.copy()

            if best_state is None:
                continue

            # --- Add node to tree ---
            node_states[n_nodes] = best_state
            node_positions[n_nodes] = best_state[self.pos_idx]
            parent_indices[n_nodes] = nearest_idx
            n_nodes += 1

            # --- Check goal ---
            ep = best_state[self.pos_idx]
            pos_err = np.sqrt(
                (ep[0] - goal_pos[0]) ** 2 + (ep[1] - goal_pos[1]) ** 2
            )
            if pos_err > goal_pos_threshold:
                continue

            heading_ok = True
            if self.heading_index is not None:
                dh = best_state[self.heading_index] - goal[self.heading_index]
                heading_ok = abs(np.arctan2(np.sin(dh), np.cos(dh))) < goal_heading_threshold

            if heading_ok:
                # Extract path by walking parent pointers
                path = []
                idx = n_nodes - 1
                while idx >= 0:
                    path.append(node_states[idx].copy())
                    idx = int(parent_indices[idx])
                path.reverse()
                return path

        return None


# ------------------------------------------------------------------ #
#  Path utilities                                                     #
# ------------------------------------------------------------------ #

def smooth_path(
    path,
    sdf_grid,
    sdf_origin,
    sdf_resolution,
    collision_margin=0.1,
    spacing=0.1,
    pos_indices=(0, 1),
    heading_index=2,
    goal_heading=None,
):
    """
    Post-process an RRT path: shortcut, densify, and recompute heading.

    1. Greedy shortcutting: skip intermediate nodes where a straight line is
       collision-free in the SDF.
    2. Densify: insert points at regular spacing (~spacing meters).
    3. Recompute heading from path tangent (direction of travel).
    4. Blend the final heading toward goal_heading if provided.

    Args:
        path: list of (D,) numpy states from RRT.
        sdf_grid: (W, H) numpy SDF array.
        sdf_origin: (2,) world coords of grid[0, 0].
        sdf_resolution: meters per cell.
        collision_margin: SDF threshold for collision-free check.
        spacing: target distance between consecutive waypoints (meters).
        pos_indices: which state dims are (x, y).
        heading_index: which state dim is heading, or None.
        goal_heading: override heading for the final waypoint (radians), or None.

    Returns:
        List of (D,) numpy states — shorter, smoother, with heading from tangent.
    """
    if len(path) <= 2:
        return path

    sdf_grid = np.asarray(sdf_grid, dtype=np.float32)
    sdf_origin = np.asarray(sdf_origin, dtype=np.float32)[:2]
    pos_idx = list(pos_indices)
    D = len(path[0])

    # --- Step 1: Greedy shortcutting ---
    shortened = [path[0].copy()]
    i = 0
    while i < len(path) - 1:
        best_j = i + 1
        for j in range(len(path) - 1, i + 1, -1):
            if _line_collision_free(
                path[i][pos_idx], path[j][pos_idx],
                sdf_grid, sdf_origin, sdf_resolution, collision_margin,
            ):
                best_j = j
                break
        shortened.append(path[best_j].copy())
        i = best_j

    # --- Step 2: Densify to regular spacing ---
    dense = [shortened[0].copy()]
    for i in range(len(shortened) - 1):
        p1 = shortened[i][pos_idx]
        p2 = shortened[i + 1][pos_idx]
        dist = np.linalg.norm(np.array(p2) - np.array(p1))
        n_seg = max(int(dist / spacing), 1)
        for j in range(1, n_seg + 1):
            t = j / n_seg
            pt = shortened[i] * (1 - t) + shortened[i + 1] * t
            dense.append(pt.copy())

    # --- Step 3: Recompute heading from tangent ---
    if heading_index is not None:
        for i in range(len(dense)):
            if i < len(dense) - 1:
                dx = dense[i + 1][pos_idx[0]] - dense[i][pos_idx[0]]
                dy = dense[i + 1][pos_idx[1]] - dense[i][pos_idx[1]]
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    dense[i][heading_index] = np.arctan2(dy, dx)
                elif i > 0:
                    dense[i][heading_index] = dense[i - 1][heading_index]
            else:
                # Last point: use previous heading or goal_heading
                dense[i][heading_index] = dense[i - 1][heading_index] if len(dense) > 1 else 0.0

        # Override final heading with goal if specified
        if goal_heading is not None:
            dense[-1][heading_index] = goal_heading
            # Smooth heading transition over last ~5 waypoints
            n_blend = min(5, len(dense) - 1)
            if n_blend > 0:
                travel_heading = dense[-(n_blend + 1)][heading_index]
                for k in range(n_blend):
                    t = (k + 1) / (n_blend + 1)
                    blended = _angle_lerp(travel_heading, goal_heading, t)
                    dense[-(n_blend - k)][heading_index] = blended

    return dense


def _line_collision_free(p1, p2, sdf_grid, sdf_origin, resolution, margin):
    """Check if a straight line segment is collision-free in the SDF."""
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    dist = np.linalg.norm(p2 - p1)
    n_checks = max(int(dist / (resolution * 0.5)), 2)
    for t in np.linspace(0.0, 1.0, n_checks):
        pt = p1 + t * (p2 - p1)
        ix = int((pt[0] - sdf_origin[0]) / resolution)
        iy = int((pt[1] - sdf_origin[1]) / resolution)
        if ix < 0 or ix >= sdf_grid.shape[0] or iy < 0 or iy >= sdf_grid.shape[1]:
            return False
        if sdf_grid[ix, iy] < margin:
            return False
    return True


def _angle_lerp(a1, a2, t):
    """Linearly interpolate between two angles, handling wraparound."""
    diff = np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))
    return a1 + t * diff


def _smooth_approach(last_wp, goal, n_points=10, spacing=0.1):
    """
    Generate a smooth approach curve from the last path waypoint to the goal
    using cubic Hermite interpolation.

    Matches position and heading at both endpoints, producing a smooth
    transition with no heading discontinuity.

    Args:
        last_wp: (D,) last waypoint [x, y, theta, ...]
        goal: (D,) goal state [x, y, theta, ...]
        n_points: minimum number of intermediate points
        spacing: target spacing in meters

    Returns:
        List of (D,) waypoints (excluding last_wp, including goal).
    """
    p0 = last_wp[:2]
    p1 = goal[:2]
    h0 = last_wp[2] if len(last_wp) > 2 else 0.0
    h1 = goal[2] if len(goal) > 2 else 0.0
    D = max(len(last_wp), len(goal))

    dist = np.linalg.norm(p1 - p0)
    if dist < 1e-4:
        return [goal.copy()]

    n = max(n_points, int(dist / spacing))

    # Tangent vectors scaled by distance (controls curve shape)
    scale = dist * 0.5
    t0 = np.array([np.cos(h0), np.sin(h0)]) * scale
    t1 = np.array([np.cos(h1), np.sin(h1)]) * scale

    # Cubic Hermite basis
    points = []
    for i in range(1, n + 1):
        s = i / n
        # Hermite basis functions
        h00 = 2 * s**3 - 3 * s**2 + 1
        h10 = s**3 - 2 * s**2 + s
        h01 = -2 * s**3 + 3 * s**2
        h11 = s**3 - s**2

        px = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
        py = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]

        # Heading from tangent of the Hermite curve
        dh00 = 6 * s**2 - 6 * s
        dh10 = 3 * s**2 - 4 * s + 1
        dh01 = -6 * s**2 + 6 * s
        dh11 = 3 * s**2 - 2 * s
        dx = dh00 * p0[0] + dh10 * t0[0] + dh01 * p1[0] + dh11 * t1[0]
        dy = dh00 * p0[1] + dh10 * t0[1] + dh01 * p1[1] + dh11 * t1[1]
        theta = np.arctan2(dy, dx) if (abs(dx) > 1e-6 or abs(dy) > 1e-6) else _angle_lerp(h0, h1, s)

        wp = np.zeros(D, dtype=np.float32)
        wp[0] = px
        wp[1] = py
        if D > 2:
            wp[2] = theta
        # Copy remaining dims from goal (e.g., velocity=0 at end)
        for d in range(3, D):
            wp[d] = last_wp[d] * (1 - s) + goal[d] * s
        points.append(wp)

    # Override final point to exact goal
    points[-1] = goal.copy()
    return points


def compute_velocity_profile(
    path,
    v_max=1.0,
    a_max=0.8,
    a_brake=1.0,
    a_lat_max=0.5,
    v_start=0.0,
    v_end=0.0,
    pos_indices=(0, 1),
    heading_index=2,
):
    """
    Compute an optimal velocity profile along a geometric path.

    Uses a forward-backward pass that respects:
    - Maximum velocity (v_max)
    - Acceleration limit (forward pass)
    - Braking limit (backward pass)
    - Curvature constraint: v <= sqrt(a_lat_max / curvature) at each point

    Args:
        path: list of (D,) numpy states with at least [x, y] positions.
        v_max: maximum allowed speed (m/s).
        a_max: maximum forward acceleration (m/s^2).
        a_brake: maximum braking deceleration (m/s^2), positive value.
        a_lat_max: maximum lateral acceleration for curvature limit (m/s^2).
        v_start: initial speed.
        v_end: final speed (typically 0 for stop-at-goal).
        pos_indices: which state dims are (x, y).
        heading_index: which state dim is heading (for curvature), or None.

    Returns:
        (N,) numpy float32 array of speeds, one per path waypoint.
    """
    path_arr = np.asarray(path, dtype=np.float32)
    N = len(path_arr)
    if N < 2:
        return np.zeros(N, dtype=np.float32)

    pos_idx = list(pos_indices)
    positions = path_arr[:, pos_idx]

    # Arc length between consecutive waypoints
    ds = np.linalg.norm(np.diff(positions, axis=0), axis=1)  # (N-1,)
    ds = np.maximum(ds, 1e-6)  # avoid division by zero

    # Curvature at each waypoint (from heading change / arc length)
    if heading_index is not None and path_arr.shape[1] > heading_index:
        headings = path_arr[:, heading_index]
        dtheta = np.abs(np.diff(headings))
        # Wrap to [0, pi]
        dtheta = np.abs(np.arctan2(np.sin(dtheta), np.cos(dtheta)))
        curvature = dtheta / ds  # (N-1,)
        # Assign curvature to each waypoint (average of neighbors)
        kappa = np.zeros(N, dtype=np.float32)
        kappa[:-1] += curvature
        kappa[1:] += curvature
        kappa[1:-1] /= 2.0
    else:
        kappa = np.zeros(N, dtype=np.float32)

    # Curvature speed limit: v <= sqrt(a_lat / kappa)
    kappa_safe = np.maximum(kappa, 1e-6)
    v_curv = np.where(kappa > 1e-4, np.sqrt(a_lat_max / kappa_safe), v_max)
    v_curv = np.minimum(v_curv, v_max)

    # Forward pass: accelerate from v_start
    v_fwd = np.zeros(N, dtype=np.float32)
    v_fwd[0] = min(v_start, v_curv[0])
    for i in range(1, N):
        # v^2 = v0^2 + 2*a*ds
        v_next = np.sqrt(max(v_fwd[i - 1] ** 2 + 2.0 * a_max * ds[i - 1], 0.0))
        v_fwd[i] = min(v_next, v_curv[i])

    # Backward pass: brake to v_end
    v_bwd = np.zeros(N, dtype=np.float32)
    v_bwd[-1] = min(v_end, v_curv[-1])
    for i in range(N - 2, -1, -1):
        v_prev = np.sqrt(max(v_bwd[i + 1] ** 2 + 2.0 * a_brake * ds[i], 0.0))
        v_bwd[i] = min(v_prev, v_curv[i])

    # Final profile: element-wise minimum of forward and backward
    v_profile = np.minimum(v_fwd, v_bwd)

    return v_profile


def track_path_goal_list(path, current_state, horizon, goal_dims=3, pos_indices=(0, 1)):
    """
    Sliding-window goal_list extraction for MPPI path tracking.

    Finds the closest point on the global path to current_state, then
    linearly interpolates the remaining path into (H+1) waypoints.
    Handles heading wraparound correctly.

    Args:
        path: list of (D,) states from global planner.
        current_state: (D,) current robot state (numpy or jax array).
        horizon: int, MPPI horizon H.
        goal_dims: number of state dims to include in goal (default 3: x, y, theta).
        pos_indices: which dims are (x, y) for proximity matching.

    Returns:
        (H+1, goal_dims) numpy float32 array.
    """
    path_arr = np.asarray(path, dtype=np.float32)
    pos = np.asarray(current_state)[list(pos_indices)]
    pos_idx = list(pos_indices)

    # Find closest point on path
    path_pos = path_arr[:, pos_idx]
    dists = np.sum((path_pos - pos) ** 2, axis=1)
    progress_idx = int(np.argmin(dists))

    # Remaining path from current position
    remaining = path_arr[progress_idx:, :goal_dims]
    n_remaining = len(remaining)
    n_needed = horizon + 1

    if n_remaining <= 1:
        point = remaining[0] if n_remaining == 1 else path_arr[-1, :goal_dims]
        return np.tile(point, (n_needed, 1))

    # Linearly interpolate remaining path to n_needed points
    old_t = np.linspace(0.0, 1.0, n_remaining)
    new_t = np.linspace(0.0, 1.0, n_needed)
    goal_list = np.zeros((n_needed, goal_dims), dtype=np.float32)

    for d in range(goal_dims):
        if d == 2:
            # Heading: unwrap before interpolation to avoid wraparound artifacts
            unwrapped = np.unwrap(remaining[:, d])
            interp = np.interp(new_t, old_t, unwrapped)
            goal_list[:, d] = np.arctan2(np.sin(interp), np.cos(interp))
        else:
            goal_list[:, d] = np.interp(new_t, old_t, remaining[:, d])

    return goal_list


def path_to_goal_list(path, horizon, goal_dims=3):
    """
    Convert a full global path to a static (H+1, D) goal_list for MPPI.

    Unlike track_path_goal_list, this does not slide — it resamples the
    entire path once. Useful for visualization or single-shot planning.

    Args:
        path: list of (D,) states.
        horizon: MPPI horizon H.
        goal_dims: dims to include.

    Returns:
        (H+1, goal_dims) numpy float32 array.
    """
    path_arr = np.asarray(path, dtype=np.float32)[:, :goal_dims]
    n_points = len(path_arr)
    n_needed = horizon + 1

    if n_points <= 1:
        point = path_arr[0] if n_points == 1 else np.zeros(goal_dims, dtype=np.float32)
        return np.tile(point, (n_needed, 1))

    old_t = np.linspace(0.0, 1.0, n_points)
    new_t = np.linspace(0.0, 1.0, n_needed)
    goal_list = np.zeros((n_needed, goal_dims), dtype=np.float32)

    for d in range(goal_dims):
        if d == 2:
            unwrapped = np.unwrap(path_arr[:, d])
            interp = np.interp(new_t, old_t, unwrapped)
            goal_list[:, d] = np.arctan2(np.sin(interp), np.cos(interp))
        else:
            goal_list[:, d] = np.interp(new_t, old_t, path_arr[:, d])

    return goal_list


# ------------------------------------------------------------------ #
#  Numpy dynamics step functions (mirrors dynamics_presets.py)         #
# ------------------------------------------------------------------ #

def make_holonomic_step(dt):
    """Numpy single-step holonomic dynamics. State: [x, y, theta]."""
    def step_fn(state, action):
        x, y, theta = state[0], state[1], state[2]
        vf, vs, omega = action[0], action[1], action[2]
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            x + (vf * c - vs * s) * dt,
            y + (vf * s + vs * c) * dt,
            theta + omega * dt,
        ], dtype=np.float32)
    return step_fn


def make_unicycle_step(dt, wheelbase=0.3):
    """Numpy single-step unicycle dynamics. State: [x, y, theta, v]."""
    def step_fn(state, action):
        x, y, theta, v = state[0], state[1], state[2], state[3]
        accel, steer = action[0], action[1]
        return np.array([
            x + v * np.cos(theta) * dt,
            y + v * np.sin(theta) * dt,
            theta + v * np.tan(steer) / wheelbase * dt,
            v + accel * dt,
        ], dtype=np.float32)
    return step_fn


def make_diff_drive_step(dt, wheel_separation=0.3):
    """Numpy single-step differential drive dynamics. State: [x, y, theta]."""
    def step_fn(state, action):
        x, y, theta = state[0], state[1], state[2]
        vl, vr = action[0], action[1]
        v = (vr + vl) / 2.0
        omega = (vr - vl) / wheel_separation
        return np.array([
            x + v * np.cos(theta) * dt,
            y + v * np.sin(theta) * dt,
            theta + omega * dt,
        ], dtype=np.float32)
    return step_fn


def make_robocasa_holonomic_step(dt_fwd=0.06, dt_side=0.06, dt_yaw=0.66):
    """Numpy single-step calibrated holonomic dynamics (RoboCasa PandaOmron)."""
    def step_fn(state, action):
        x, y, theta = state[0], state[1], state[2]
        vf, vs, omega = action[0], action[1], action[2]
        c, s = np.cos(theta), np.sin(theta)
        dx_body = vf * dt_fwd
        dy_body = vs * dt_side
        return np.array([
            x + (dx_body * c - dy_body * s),
            y + (dx_body * s + dy_body * c),
            theta + omega * dt_yaw,
        ], dtype=np.float32)
    return step_fn


# Registry: matches dynamics_presets.DYNAMICS_REGISTRY keys
GLOBAL_DYNAMICS_REGISTRY = {
    'holonomic': {
        'factory': lambda dt, **kw: make_holonomic_step(dt),
        'state_dims': 3,
        'action_dims': 3,
    },
    'robocasa_holonomic': {
        'factory': lambda dt, **kw: make_robocasa_holonomic_step(
            dt_fwd=kw.get('dt_fwd', 0.06),
            dt_side=kw.get('dt_side', 0.06),
            dt_yaw=kw.get('dt_yaw', 0.66),
        ),
        'state_dims': 3,
        'action_dims': 3,
    },
    'unicycle': {
        'factory': lambda dt, **kw: make_unicycle_step(dt, kw.get('wheelbase', 0.3)),
        'state_dims': 4,
        'action_dims': 2,
    },
    'diff_drive': {
        'factory': lambda dt, **kw: make_diff_drive_step(dt, kw.get('wheel_separation', 0.3)),
        'state_dims': 3,
        'action_dims': 2,
    },
}


# ------------------------------------------------------------------ #
#  Convenience API                                                    #
# ------------------------------------------------------------------ #

def plan_global_path(
    start,
    goal,
    dynamics,
    obstacle_map,
    dt=0.05,
    action_limit=1.0,
    max_iters=5000,
    extend_steps=10,
    n_action_samples=5,
    goal_bias=0.3,
    goal_pos_threshold=0.2,
    goal_heading_threshold=0.5,
    collision_margin=0.05,
    seed=42,
    max_retries=5,
    **dynamics_kwargs,
):
    """
    One-call convenience function: plan a kinodynamically feasible global path.

    Uses the same dynamics string as create_planner() so both layers share
    the same model. Also accepts a custom step_fn callable.

    Automatically retries with different random seeds if the first attempt
    fails (standard for sampling-based planners).

    Args:
        start: (D,) start state.
        goal: (D,) goal state.
        dynamics: 'holonomic', 'unicycle', 'diff_drive', 'robocasa_holonomic',
            or a callable step_fn(state, action) -> next_state.
        obstacle_map: ObstacleMap instance.
        dt: timestep (used by generic dynamics; robocasa_holonomic uses its own dt_*).
        action_limit: symmetric action bounds [-limit, limit].
        max_iters: maximum RRT iterations.
        extend_steps: dynamics steps per tree extension. Increase for small dt.
        n_action_samples: random actions to try per extension.
        goal_bias: probability of sampling goal as target.
        goal_pos_threshold: position tolerance (meters).
        goal_heading_threshold: heading tolerance (radians).
        collision_margin: min SDF distance for collision-free.
        seed: random seed for first attempt.
        max_retries: number of attempts with different seeds (default 5).
        **dynamics_kwargs: passed to dynamics factory (e.g., wheelbase, dt_fwd).

    Returns:
        List of (D,) numpy states from start to goal, or None if all attempts fail.
    """
    if isinstance(dynamics, str):
        if dynamics not in GLOBAL_DYNAMICS_REGISTRY:
            raise ValueError(
                f"Unknown dynamics '{dynamics}'. "
                f"Available: {list(GLOBAL_DYNAMICS_REGISTRY.keys())}. "
                f"Or pass a callable step_fn."
            )
        entry = GLOBAL_DYNAMICS_REGISTRY[dynamics]
        step_fn = entry['factory'](dt, **dynamics_kwargs)
        state_dims = entry['state_dims']
        action_dims = entry['action_dims']
    elif callable(dynamics):
        step_fn = dynamics
        state_dims = dynamics_kwargs.get('state_dims')
        action_dims = dynamics_kwargs.get('action_dims')
        if state_dims is None or action_dims is None:
            raise ValueError(
                "When passing a custom step_fn, must also provide "
                "state_dims and action_dims in kwargs."
            )
    else:
        raise TypeError(f"dynamics must be str or callable, got {type(dynamics)}")

    action_low = np.full(action_dims, -action_limit, dtype=np.float32)
    action_high = np.full(action_dims, action_limit, dtype=np.float32)

    rrt = KinodynamicRRT(
        step_fn=step_fn,
        state_dims=state_dims,
        action_dims=action_dims,
        action_low=action_low,
        action_high=action_high,
    )

    sdf_np = np.asarray(obstacle_map.sdf, dtype=np.float32)
    origin_np = np.asarray(obstacle_map.origin, dtype=np.float32)

    for attempt in range(max_retries):
        path = rrt.plan(
            start=start,
            goal=goal,
            sdf_grid=sdf_np,
            sdf_origin=origin_np,
            sdf_resolution=float(obstacle_map.resolution),
            max_iters=max_iters,
            extend_steps=extend_steps,
            n_action_samples=n_action_samples,
            goal_bias=goal_bias,
            goal_pos_threshold=goal_pos_threshold,
            goal_heading_threshold=goal_heading_threshold,
            collision_margin=collision_margin,
            seed=seed + attempt,
        )
        if path is not None:
            # Post-process: shortcut, densify, recompute heading
            goal_state = np.asarray(goal, dtype=np.float32)
            heading_idx = 2 if state_dims >= 3 else None
            goal_heading = float(goal_state[heading_idx]) if heading_idx is not None else None
            path = smooth_path(
                path,
                sdf_np,
                origin_np,
                float(obstacle_map.resolution),
                collision_margin=collision_margin,
                spacing=0.1,
                heading_index=heading_idx,
                goal_heading=goal_heading,
            )
            # Remove last 5 waypoints, append exact goal, re-densify + re-smooth
            n_trim = min(5, len(path) - 2)
            path = path[:-n_trim]
            goal_wp = goal_state[:len(path[0])].copy()
            path.append(goal_wp)
            # Re-densify the last segment (path[-2] → path[-1])
            p_prev = path[-2]
            p_goal = path[-1]
            dist = np.linalg.norm(p_goal[:2] - p_prev[:2])
            n_insert = max(int(dist / 0.1), 1)
            new_tail = []
            for j in range(1, n_insert):
                t = j / n_insert
                wp = p_prev * (1 - t) + p_goal * t
                # Heading from tangent
                if heading_idx is not None:
                    wp[heading_idx] = _angle_lerp(p_prev[heading_idx], goal_heading, t)
                new_tail.append(wp)
            # Insert before the goal
            path = path[:-1] + new_tail + [goal_wp]
            # Smooth heading over the final portion
            if heading_idx is not None:
                n_blend = min(10, len(path) - 1)
                for k in range(n_blend):
                    i = len(path) - 1 - n_blend + k
                    if i < 1 or i >= len(path) - 1:
                        continue
                    t = (k + 1) / (n_blend + 1)
                    dx = path[i + 1][0] - path[i][0]
                    dy = path[i + 1][1] - path[i][1]
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                        tangent_h = np.arctan2(dy, dx)
                        path[i][heading_idx] = _angle_lerp(tangent_h, goal_heading, t)
                path[-1][heading_idx] = goal_heading

            # Compute and attach velocity profile
            v_profile = compute_velocity_profile(path)
            # Ensure all waypoints are at least 4D: [x, y, θ, v]
            for i in range(len(path)):
                wp = path[i]
                if len(wp) < 4:
                    path[i] = np.concatenate([wp, [v_profile[i]]])
                else:
                    path[i] = wp.copy()
                    path[i][3] = v_profile[i]

            return path

    return None
