"""
FastAPI backend for MPPI planner visualization.

Serves a web UI and provides an SSE streaming endpoint that runs
a collision-aware MPPI planner step-by-step, streaming results to the browser.

Run from repo root:
    uv run python viz/app.py
"""

import asyncio
import json
import math
import os
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from car_dynamics.controllers_jax import (
    ObstacleMap,
    create_planner,
    plan_global_path,
    track_path_goal_list,
    hybrid_astar_global_path,
)

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="MPPI Planner Visualizer")

VIZ_DIR = Path(__file__).resolve().parent
STATIC_DIR = VIZ_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIST = VIZ_DIR / "frontend" / "dist"

# Serve React build assets if available
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

# Fallback: serve old static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Dynamics metadata
# ---------------------------------------------------------------------------

DYNAMICS_INFO = {
    "holonomic": {"state_dims": 3, "action_dims": 3, "kwargs": {}},
    "unicycle": {"state_dims": 4, "action_dims": 2, "kwargs": {"wheelbase": 0.3}},
    "diff_drive": {"state_dims": 3, "action_dims": 2, "kwargs": {"wheel_separation": 0.3}},
    "robocasa_holonomic": {
        "state_dims": 3,
        "action_dims": 3,
        "kwargs": {"dt_fwd": 0.06, "dt_side": 0.06, "dt_yaw": 0.66},
    },
}

# ---------------------------------------------------------------------------
# Forward-simulation helpers (numpy, single state)
# ---------------------------------------------------------------------------


def forward_step(dynamics: str, state: np.ndarray, action: np.ndarray,
                 dt: float, **kwargs) -> np.ndarray:
    """Euler-integrate one step for the given dynamics type."""
    if dynamics == "holonomic":
        x, y, theta = state[0], state[1], state[2]
        vf, vs, omega = action[0], action[1], action[2]
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            x + (vf * c - vs * s) * dt,
            y + (vf * s + vs * c) * dt,
            theta + omega * dt,
        ], dtype=np.float32)

    elif dynamics == "robocasa_holonomic":
        x, y, theta = state[0], state[1], state[2]
        vf, vs, omega = action[0], action[1], action[2]
        dt_fwd = kwargs.get("dt_fwd", 0.06)
        dt_side = kwargs.get("dt_side", 0.06)
        dt_yaw = kwargs.get("dt_yaw", 0.66)
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            x + (vf * dt_fwd * c - vs * dt_side * s),
            y + (vf * dt_fwd * s + vs * dt_side * c),
            theta + omega * dt_yaw,
        ], dtype=np.float32)

    elif dynamics == "unicycle":
        x, y, theta, v = state[0], state[1], state[2], state[3]
        accel, steer = action[0], action[1]
        wheelbase = kwargs.get("wheelbase", 0.3)
        return np.array([
            x + v * math.cos(theta) * dt,
            y + v * math.sin(theta) * dt,
            theta + v * math.tan(steer) / wheelbase * dt,
            v + accel * dt,
        ], dtype=np.float32)

    elif dynamics == "diff_drive":
        x, y, theta = state[0], state[1], state[2]
        vl, vr = action[0], action[1]
        wheel_sep = kwargs.get("wheel_separation", 0.3)
        v = (vr + vl) / 2.0
        omega = (vr - vl) / wheel_sep
        return np.array([
            x + v * math.cos(theta) * dt,
            y + v * math.sin(theta) * dt,
            theta + omega * dt,
        ], dtype=np.float32)

    else:
        raise ValueError(f"Unknown dynamics: {dynamics}")


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse(event: str, data) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _downsample_grid(grid: np.ndarray, max_dim: int = 100) -> list:
    """Downsample a 2D grid to at most max_dim x max_dim and return as nested list."""
    h, w = grid.shape
    step_h = max(1, h // max_dim)
    step_w = max(1, w // max_dim)
    small = grid[::step_h, ::step_w]
    return small.tolist()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    if FRONTEND_DIST.exists():
        return FileResponse(str(FRONTEND_DIST / "index.html"))
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/dynamics")
async def dynamics_info():
    return DYNAMICS_INFO


@app.get("/api/plan")
async def plan_sse(
    dynamics: str = Query("holonomic"),
    boxes: str = Query("[]"),
    circles: str = Query("[]"),
    bounds: str = Query("[-5,-5,5,5]"),
    start: str = Query("[0,0,0]"),
    goal: str = Query("[4,4,0]"),
    resolution: float = Query(0.05),
    inflate_radius: float = Query(0.15),
    dt: float = Query(0.05),
    n_rollouts: int = Query(2048),
    sigma: float = Query(0.15),
    steer_sigma: float = Query(0.0),
    lam: float = Query(0.02),
    horizon_knots: int = Query(12),
    num_intermediate: int = Query(6),
    goal_pos_weight: float = Query(5000.0),
    goal_heading_weight: float = Query(200.0),
    goal_vel_weight: float = Query(0.0),
    collision_weight: float = Query(30.0),
    collision_decay: float = Query(3.0),
    collision_inside_weight: float = Query(300.0),
    smoothness_weight: float = Query(10.0),
    terminal_weight: float = Query(10.0),
    control_reg_weight: float = Query(0.1),
    sign_change_weight: float = Query(0.0),
    goal_reach_reward: float = Query(0.0),
    goal_reach_threshold: float = Query(0.1),
    replan_interval: int = Query(1),
    stop_threshold: float = Query(0.08),
    use_global_planner: bool = Query(True),
    global_planner_type: str = Query("rrt"),
    gamma_sigma: float = Query(0.0),
    cov_floor: float = Query(0.0),
    backend: str = Query("spline"),
    max_steps: int = Query(300),
    goal_tolerance: float = Query(0.05),
    # Dynamics-specific kwargs
    wheelbase: float = Query(0.3),
    wheel_separation: float = Query(0.3),
    dt_fwd: float = Query(0.06),
    dt_side: float = Query(0.06),
    dt_yaw: float = Query(0.66),
):
    async def generate():
        try:
            # ---- Parse JSON params ----
            boxes_list = json.loads(boxes)
            circles_list = json.loads(circles)
            bounds_arr = json.loads(bounds)
            start_arr = np.array(json.loads(start), dtype=np.float32)
            goal_arr = np.array(json.loads(goal), dtype=np.float32)

            min_xy = np.array(bounds_arr[:2], dtype=np.float32)
            max_xy = np.array(bounds_arr[2:], dtype=np.float32)

            # ---- Build dynamics kwargs ----
            dyn_kwargs = {}
            if dynamics == "unicycle":
                dyn_kwargs["wheelbase"] = wheelbase
            elif dynamics == "diff_drive":
                dyn_kwargs["wheel_separation"] = wheel_separation
            elif dynamics == "robocasa_holonomic":
                dyn_kwargs["dt_fwd"] = dt_fwd
                dyn_kwargs["dt_side"] = dt_side
                dyn_kwargs["dt_yaw"] = dt_yaw

            # ---- Build ObstacleMap ----
            # Convert boxes from [x1,y1,x2,y2] flat format to (min_xy, max_xy) tuples
            parsed_boxes = []
            for b in boxes_list:
                if len(b) == 4:
                    parsed_boxes.append(([b[0], b[1]], [b[2], b[3]]))
                else:
                    parsed_boxes.append(b)

            obstacle_map = None
            if parsed_boxes:
                obstacle_map = ObstacleMap.from_boxes(
                    parsed_boxes,
                    bounds=(min_xy, max_xy),
                    resolution=resolution,
                    inflate_radius=inflate_radius,
                )

            # Combine with circles if present
            if circles_list:
                centers = np.array([c[:2] for c in circles_list], dtype=np.float32)
                radii = np.array([c[2] for c in circles_list], dtype=np.float32)
                circle_map = ObstacleMap.from_circles(
                    centers, radii,
                    bounds=(min_xy, max_xy),
                    resolution=resolution,
                )
                if obstacle_map is not None:
                    # Combine SDFs: element-wise min
                    combined_sdf = jnp.minimum(obstacle_map.sdf, circle_map.sdf)
                    obstacle_map = ObstacleMap(combined_sdf, obstacle_map.origin, obstacle_map.resolution)
                else:
                    obstacle_map = circle_map

            # Fallback: empty obstacle map
            if obstacle_map is None:
                nx = int(np.ceil((max_xy[0] - min_xy[0]) / resolution))
                ny = int(np.ceil((max_xy[1] - min_xy[1]) / resolution))
                sdf = jnp.full((nx, ny), 999.0)
                obstacle_map = ObstacleMap(sdf, jnp.array(min_xy), resolution)

            # ---- Yield SDF ----
            sdf_np = np.asarray(obstacle_map.sdf)
            sdf_downsampled = _downsample_grid(sdf_np, max_dim=100)
            sdf_origin = np.asarray(obstacle_map.origin).tolist()
            yield _sse("sdf", {
                "grid": sdf_downsampled,
                "origin": sdf_origin,
                "resolution": obstacle_map.resolution,
                "full_shape": list(sdf_np.shape),
            })
            await asyncio.sleep(0)

            # ---- Create planner ----
            # Build per-action sigma if steer_sigma is specified
            dyn_info = DYNAMICS_INFO[dynamics]
            action_dims = dyn_info["action_dims"]
            if steer_sigma > 0:
                sigma_arr = [sigma] * action_dims
                sigma_arr[-1] = steer_sigma  # last action dim is steer/omega
                planner_sigma = sigma_arr
            else:
                planner_sigma = sigma

            planner, running_params = create_planner(
                dynamics=dynamics,
                obstacle_map=obstacle_map,
                dt=dt,
                n_rollouts=n_rollouts,
                horizon_knots=horizon_knots,
                num_intermediate=num_intermediate,
                sigma=planner_sigma,
                lam=lam,
                collision_weight=collision_weight if use_global_planner else max(collision_weight, 500.0),
                collision_decay=collision_decay,
                collision_inside_weight=collision_inside_weight if use_global_planner else max(collision_inside_weight, 100000.0),
                goal_pos_weight=goal_pos_weight,
                goal_heading_weight=goal_heading_weight,
                goal_vel_weight=goal_vel_weight,
                smoothness_weight=smoothness_weight,
                terminal_weight=terminal_weight,
                control_reg_weight=control_reg_weight,
                sign_change_weight=sign_change_weight,
                goal_reach_reward=goal_reach_reward,
                goal_reach_threshold=goal_reach_threshold,
                gamma_sigma=gamma_sigma,
                cov_floor=cov_floor,
                adaptive_beta=(backend == 'torch'),
                backend=backend,
                **dyn_kwargs,
            )

            # ---- Pad start state to match dynamics state_dims ----
            state_dims = dyn_info["state_dims"]
            state = np.zeros(state_dims, dtype=np.float32)
            state[:min(len(start_arr), state_dims)] = start_arr[:state_dims]

            goal_state = np.zeros(state_dims, dtype=np.float32)
            goal_state[:min(len(goal_arr), state_dims)] = goal_arr[:state_dims]

            # ---- Check if global planning needed ----
            # Skip RRT if no real obstacles (SDF is all free space)
            has_obstacles = parsed_boxes or circles_list
            dist_to_goal = np.linalg.norm(state[:2] - goal_state[:2])
            global_path = None
            if dist_to_goal > 1.0 and has_obstacles and use_global_planner:
                if global_planner_type == "hybrid_astar":
                    global_path = await asyncio.to_thread(
                        hybrid_astar_global_path,
                        start=state,
                        goal=goal_state,
                        dynamics=dynamics,
                        obstacle_map=obstacle_map,
                        dt=dt,
                        v_default=0.5,
                        extend_steps=8,
                        grid_xy_res=0.2,
                        goal_pos_threshold=0.3,
                        **dyn_kwargs,
                    )
                else:
                    global_path = await asyncio.to_thread(
                        plan_global_path,
                        start=state,
                        goal=goal_state,
                        dynamics=dynamics,
                        obstacle_map=obstacle_map,
                        dt=dt,
                        **dyn_kwargs,
                    )
                if global_path is not None:
                    path_list = [s.tolist() for s in global_path]
                    yield _sse("global_path", {"path": path_list})
                    await asyncio.sleep(0)

            # ---- MPPI loop ----
            horizon = planner.H
            trajectory_history = []
            actions_history = []
            start_arr_full = state.copy()

            # Determine goal_dims: use 4 for unicycle (state_dims>=4) to always
            # include velocity=0 target, preventing forward/backward oscillation
            use_vel = state_dims >= 4 and goal_vel_weight > 0
            mppi_goal_dims = 4 if use_vel else 3

            step_i = 0
            a_mean_seq = None  # cached action sequence from last MPPI call
            exec_idx = 0       # index into a_mean_seq

            while step_i < max_steps:
                # Replan if needed
                if a_mean_seq is None or exec_idx >= replan_interval:
                    # Build goal_list
                    if global_path is not None:
                        goal_list = track_path_goal_list(
                            global_path, state, horizon, goal_dims=mppi_goal_dims,
                        )
                    else:
                        goal_tile = goal_state[:mppi_goal_dims]
                        goal_list = np.tile(goal_tile, (horizon + 1, 1))

                    goal_list_jax = jnp.array(goal_list)

                    # Run full MPPI optimization
                    action, running_params, step_info = await asyncio.to_thread(
                        planner, state, goal_list_jax, running_params, ()
                    )
                    running_params = planner.feed_hist(running_params, state, action)

                    # Cache the full mean action sequence for multi-step execution
                    if "a_mean_jnp" in step_info and step_info["a_mean_jnp"] is not None:
                        a_mean_seq = np.asarray(step_info["a_mean_jnp"])
                    else:
                        a_mean_seq = np.asarray(action)[None, :]  # fallback: single action
                    exec_idx = 0

                    # Extract viz data (only on replan steps)
                    pred_traj = None
                    if "trajectory" in step_info and step_info["trajectory"] is not None:
                        pred_traj = np.asarray(step_info["trajectory"]).tolist()
                    sampled_trajs = None
                    if "sampled_trajectories" in step_info and step_info["sampled_trajectories"] is not None:
                        st = np.asarray(step_info["sampled_trajectories"])
                        sampled_trajs = []
                        for k in range(st.shape[1]):
                            sampled_trajs.append(st[:, k, :2].tolist())

                # Execute action from cached sequence
                act_idx = min(exec_idx, len(a_mean_seq) - 1)
                action_np = a_mean_seq[act_idx]

                # Stop-and-hold: when within stop_threshold, override action to zero
                current_dist = float(np.linalg.norm(state[:2] - goal_state[:2]))
                if current_dist < stop_threshold:
                    action_np = np.zeros_like(action_np)

                actions_history.append(action_np.copy())
                exec_idx += 1

                # Forward-simulate state
                state = forward_step(dynamics, state, action_np, dt, **dyn_kwargs)
                trajectory_history.append(state.tolist())

                # Compute errors
                pos_err = float(np.linalg.norm(state[:2] - goal_state[:2]))
                heading_err = float(abs(math.atan2(
                    math.sin(state[2] - goal_state[2]),
                    math.cos(state[2] - goal_state[2]),
                )))

                # Check if predicted trajectory triggers goal reach reward
                goal_reward_triggered = False
                if pred_traj is not None and goal_reach_reward > 0:
                    for pt in pred_traj:
                        d = math.sqrt((pt[0] - goal_state[0])**2 + (pt[1] - goal_state[1])**2)
                        if d < goal_reach_threshold:
                            goal_reward_triggered = True
                            break

                yield _sse("step", {
                    "step": step_i,
                    "state": state.tolist(),
                    "action": action_np.tolist(),
                    "pos_error": pos_err,
                    "heading_error": heading_err,
                    "predicted_trajectory": pred_traj if exec_idx == 1 else None,
                    "sampled_trajectories": sampled_trajs if exec_idx == 1 else None,
                    "goal_reward_triggered": goal_reward_triggered,
                })
                await asyncio.sleep(0)

                step_i += 1
                if pos_err < goal_tolerance:
                    break

            # Compute SGF-smoothed baseline trajectory
            actions_arr = np.array(actions_history)
            smoothed_traj = None
            if len(actions_arr) > 7:
                from scipy.signal import savgol_filter
                window = min(15, len(actions_arr) - 1)
                if window % 2 == 0:
                    window -= 1
                window = max(3, window)
                smoothed_actions = np.copy(actions_arr)
                for d in range(actions_arr.shape[1]):
                    smoothed_actions[:, d] = savgol_filter(
                        actions_arr[:, d], window, min(3, window - 1),
                    )
                # Re-simulate with smoothed actions
                sim_state = np.array(start_arr_full, dtype=np.float32)
                smoothed_traj = [sim_state.tolist()]
                for a in smoothed_actions:
                    sim_state = forward_step(dynamics, sim_state, a, dt, **dyn_kwargs)
                    smoothed_traj.append(sim_state.tolist())

            yield _sse("done", {
                "steps": step_i + 1,
                "final_state": state.tolist(),
                "final_pos_error": float(np.linalg.norm(state[:2] - goal_state[:2])),
                "trajectory": trajectory_history,
                "smoothed_trajectory": smoothed_traj,
            })

        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
