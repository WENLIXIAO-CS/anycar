"""
Remote MPPI navigation planner server.

Accepts collision geometry and navigation goals via Portal RPC,
returns collision-free trajectories for holonomic mobile bases.

Usage:
    cd /home/lecar-lab/anycar
    uv run python serve_mppi_planner.py --port 18700

Client connects via:
    CAP_MPPI_HOST=<ip> CAP_MPPI_PORT=18700
"""

import argparse
import logging
import time

import numpy as np
import portal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MPPI] %(message)s")
logger = logging.getLogger(__name__)


class MPPIPlannerServer:
    def __init__(self, port: int, warmup: bool = True):
        self._server = portal.Server(port, logging=False)
        self._server.bind("health_check", self._handle_health_check)
        self._server.bind("update_world", self._handle_update_world)
        self._server.bind("plan_trajectory", self._handle_plan_trajectory)

        self._planner = None
        self._obstacle_map = None
        self._port = port

        if warmup:
            self._warmup_jit()

    def _warmup_jit(self):
        """Run a dummy planning cycle to trigger JAX JIT compilation."""
        logger.info("Warming up JAX JIT (first compile takes ~10-30s)...")
        from car_dynamics.controllers_jax import create_planner, ObstacleMap

        # Warmup with 100×100 fixed grid (same shape as production)
        dummy_map = ObstacleMap.from_boxes(
            [([0.0, 0.0], [0.5, 0.5])],
            bounds=([-3.1, -3.1], [3.1, 3.1]),
            resolution=6.2 / 100,
        )
        planner, rp = create_planner(
            dynamics="robocasa_holonomic",
            obstacle_map=dummy_map,
            n_rollouts=2048,
            dt=1.0,
            horizon_knots=10,
            num_intermediate=8,
            dt_fwd=0.06,
            dt_side=0.06,
            dt_yaw=0.66,
        )
        import jax.numpy as jnp

        state = jnp.array([0.0, 0.0, 0.0])
        goal = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (planner.H + 1, 1))
        _ = planner(state, goal, rp, ())
        logger.info("JIT warmup complete.")

    def _load_dynamics_params(self):
        """Load dynamics params from forge's calibration file, or use defaults."""
        import json
        params_path = "/home/lecar-lab/forge/tools/dynamics_params.json"
        defaults = {"dt_fwd": 0.06, "dt_side": 0.06, "dt_yaw": 0.66}
        try:
            with open(params_path) as f:
                params = json.load(f)
            logger.info("Loaded dynamics params from %s: %s", params_path, params)
            defaults.update(params)
        except FileNotFoundError:
            logger.info("No dynamics_params.json — using defaults: %s", defaults)
        return defaults

    def _handle_health_check(self):
        return True

    def _handle_update_world(self, payload):
        """
        Receive collision geometry and rebuild ObstacleMap + planner.

        payload:
            boxes: list of [center_x, center_y, half_w, half_h]
            bounds_min: [x, y]
            bounds_max: [x, y]
            resolution: float (default 0.02)
            inflate_radius: float (default 0.15)
        """
        from car_dynamics.controllers_jax import ObstacleMap, create_planner

        boxes_raw = payload["boxes"]
        bounds = (
            np.array(payload["bounds_min"], dtype=np.float32),
            np.array(payload["bounds_max"], dtype=np.float32),
        )
        resolution = float(payload.get("resolution", 0.02))
        inflate_radius = float(payload.get("inflate_radius", 0.15))

        # Convert [cx, cy, hw, hh] to (center_xy, half_ext_xy, yaw=0) format
        boxes = []
        for b in boxes_raw:
            center = np.array([b[0], b[1]], dtype=np.float32)
            half_ext = np.array([b[2], b[3]], dtype=np.float32)
            boxes.append((center, half_ext, 0.0))

        # Pad bounds to a fixed grid size so JAX never needs to retrace
        # (JIT recompiles whenever array shapes change)
        FIXED_GRID = 100  # always 100×100 grid
        span_x = float(bounds[1][0] - bounds[0][0])
        span_y = float(bounds[1][1] - bounds[0][1])
        max_span = max(span_x, span_y)
        fixed_res = max_span / FIXED_GRID
        # Expand bounds to square
        center = (bounds[0] + bounds[1]) / 2
        half = max_span / 2 + 0.1
        bounds = (center - half, center + half)
        resolution = fixed_res
        logger.info("Grid: %.1fm × %.1fm at %.3fm/cell = %dx%d (fixed)",
                    max_span, max_span, resolution, FIXED_GRID, FIXED_GRID)

        t0 = time.time()
        self._obstacle_map = ObstacleMap.from_boxes(
            boxes, bounds, resolution, inflate_radius
        )
        self._planner, _ = create_planner(
            dynamics="robocasa_holonomic",
            obstacle_map=self._obstacle_map,
            dt=1.0,  # ignored — robocasa_holonomic uses per-axis dt internally
            n_rollouts=2048,
            horizon_knots=10,
            num_intermediate=8,
            sigma=0.15,
            lam=0.005,
            collision_weight=30.0,
            collision_decay=3.0,
            collision_inside_weight=300.0,
            goal_pos_weight=5000.0,
            goal_heading_weight=1.0,
            smoothness_weight=10.0,
            # Load calibrated dynamics params
            **self._load_dynamics_params(),
        )
        dt = time.time() - t0
        logger.info(
            "World updated: %d obstacles, SDF %s, %.2fs",
            len(boxes_raw),
            list(self._obstacle_map.shape),
            dt,
        )
        return {"n_obstacles": len(boxes_raw), "sdf_shape": list(self._obstacle_map.shape)}

    def _handle_plan_trajectory(self, payload):
        """
        Plan collision-free trajectory from current state to goal.

        For targets beyond local MPPI reach, a kinodynamic RRT global planner
        produces intermediate waypoints that MPPI tracks via a sliding window.

        payload:
            current_pos: [x, y]
            current_yaw: float (radians)
            target_pos: [x, y]
            target_yaw: float (radians)
            max_steps: int (default 400)
            goal_tolerance: float (default 0.05 meters)
            yaw_tolerance: float (default 0.1 radians)

        Returns:
            status: "Success" | "Timeout" | "NoPath" | "Error"
            trajectory: (T+1, 3) -- [x, y, yaw] per step
            actions: (T, 3) -- [v_fwd, v_side, omega] per step
            n_steps: int
            final_pos_error: float
            final_yaw_error: float
            used_global_planner: bool
        """
        if self._planner is None:
            return {"status": "Error", "reason": "Call update_world first"}

        import jax.numpy as jnp
        from car_dynamics.controllers_jax.global_planner import (
            plan_global_path, track_path_goal_list,
        )

        current_pos = np.array(payload["current_pos"], dtype=np.float32)
        current_yaw = float(payload["current_yaw"])
        target_pos = np.array(payload["target_pos"], dtype=np.float32)
        target_yaw = float(payload["target_yaw"])
        max_steps = int(payload.get("max_steps", 10))
        goal_tol = float(payload.get("goal_tolerance", 0.05))
        yaw_tol = float(payload.get("yaw_tolerance", 0.1))

        # Load calibrated dynamics params
        _dp = self._load_dynamics_params()
        DT_FWD = _dp["dt_fwd"]
        DT_SIDE = _dp["dt_side"]
        DT_YAW = _dp["dt_yaw"]

        H = self._planner.H
        start_state = np.array([current_pos[0], current_pos[1], current_yaw])
        goal_state = np.array([target_pos[0], target_pos[1], target_yaw])

        # Decide whether to use global planner based on distance vs horizon reach
        dist_to_goal = float(np.linalg.norm(current_pos - target_pos))
        horizon_reach = H * DT_FWD  # max distance MPPI can "see"
        use_global = dist_to_goal > horizon_reach * 0.5

        global_path = None
        if use_global:
            t_rrt = time.time()
            global_path = plan_global_path(
                start=start_state,
                goal=goal_state,
                dynamics="robocasa_holonomic",
                obstacle_map=self._obstacle_map,
                dt_fwd=DT_FWD,
                dt_side=DT_SIDE,
                dt_yaw=DT_YAW,
                max_iters=5000,
                extend_steps=5,
                goal_pos_threshold=0.15,
                goal_heading_threshold=0.5,
                collision_margin=0.05,
            )
            dt_rrt = time.time() - t_rrt
            if global_path is None:
                logger.warning("Global RRT failed to find path (%.2fs)", dt_rrt)
                return {
                    "status": "NoPath",
                    "trajectory": np.array([start_state]),
                    "actions": np.zeros((0, 3)),
                    "n_steps": 0,
                    "final_pos_error": dist_to_goal,
                    "final_yaw_error": float("inf"),
                    "used_global_planner": True,
                }
            logger.info(
                "Global RRT: %d waypoints in %.2fs (dist=%.2fm)",
                len(global_path), dt_rrt, dist_to_goal,
            )

        state = jnp.array(start_state)
        trajectory = [np.array(state)]
        actions_list = []
        running_params = self._planner.get_init_params()
        t0 = time.time()

        for step in range(max_steps):
            # Build goal_list: sliding window on global path, or tiled single goal
            if global_path is not None:
                gl_np = track_path_goal_list(
                    global_path, np.asarray(state), H, goal_dims=3,
                )
                goal_list = jnp.array(gl_np)
            else:
                goal = jnp.array(goal_state)
                goal_list = jnp.tile(goal, (H + 1, 1))

            action, running_params, info = self._planner(
                state, goal_list, running_params, ()
            )
            running_params = self._planner.feed_hist(running_params, state, action)
            actions_list.append(np.array(action))

            # Forward-simulate one step (calibrated per-axis dynamics)
            vf, vs, omega = float(action[0]), float(action[1]), float(action[2])
            theta = float(state[2])
            dx_body = vf * DT_FWD
            dy_body = vs * DT_SIDE
            x = float(state[0]) + (dx_body * np.cos(theta) - dy_body * np.sin(theta))
            y = float(state[1]) + (dx_body * np.sin(theta) + dy_body * np.cos(theta))
            theta_new = theta + omega * DT_YAW
            state = jnp.array([x, y, theta_new])
            trajectory.append(np.array(state))

            # Check convergence
            pos_err = float(np.linalg.norm(np.array([x, y]) - target_pos))
            yaw_err = abs(
                np.arctan2(
                    np.sin(theta_new - target_yaw), np.cos(theta_new - target_yaw)
                )
            )
            if pos_err < goal_tol and yaw_err < yaw_tol:
                break

        elapsed = time.time() - t0
        status = "Success" if (pos_err < goal_tol and yaw_err < yaw_tol) else "Timeout"
        n_steps = len(actions_list)

        logger.info(
            "Plan: %s in %d steps (%.2fs), pos_err=%.3fm, yaw_err=%.3frad, global=%s",
            status,
            n_steps,
            elapsed,
            pos_err,
            yaw_err,
            use_global,
        )

        traj_np = np.stack(trajectory)
        acts_np = np.stack(actions_list)

        # Smooth actions with exponential moving average
        if len(acts_np) > 1:
            alpha = 0.4  # smoothing factor (0=full smooth, 1=no smooth)
            smoothed = np.copy(acts_np)
            for i in range(1, len(smoothed)):
                smoothed[i] = alpha * smoothed[i] + (1 - alpha) * smoothed[i - 1]
            acts_np = smoothed

            # Re-derive trajectory from smoothed actions
            traj_smooth = [traj_np[0]]
            state_s = traj_np[0].copy()
            for a in acts_np:
                theta_s = state_s[2]
                dx_b = float(a[0]) * DT_FWD
                dy_b = float(a[1]) * DT_SIDE
                state_s = np.array([
                    state_s[0] + (dx_b * np.cos(theta_s) - dy_b * np.sin(theta_s)),
                    state_s[1] + (dx_b * np.sin(theta_s) + dy_b * np.cos(theta_s)),
                    state_s[2] + a[2] * DT_YAW,
                ])
                traj_smooth.append(state_s)
            traj_np = np.stack(traj_smooth)

        return {
            "status": status,
            "trajectory": traj_np,
            "actions": acts_np,
            "n_steps": n_steps,
            "final_pos_error": float(pos_err),
            "final_yaw_error": float(yaw_err),
            "used_global_planner": use_global,
        }

    def serve(self):
        logger.info("MPPI planner server starting on port %d", self._port)
        self._server.start()


def main():
    parser = argparse.ArgumentParser(description="Remote MPPI navigation planner")
    parser.add_argument("--port", type=int, default=18700, help="Portal RPC port")
    parser.add_argument(
        "--no-warmup", action="store_true", help="Skip JIT warmup on startup"
    )
    args = parser.parse_args()

    server = MPPIPlannerServer(port=args.port, warmup=not args.no_warmup)
    server.serve()


if __name__ == "__main__":
    main()
