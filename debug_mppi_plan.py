"""
Debug MPPI planner — visualize obstacle map + planned trajectory.

Usage:
    # Connect to running MPPI server:
    uv run python debug_mppi_plan.py --port 18700

    # Standalone (no server needed):
    uv run python debug_mppi_plan.py --standalone
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


def plan_standalone(
    start=(0.0, 0.0, 0.0),
    goal=(1.5, 0.5, 0.0),
    boxes=None,
    bounds=((-2, -2), (3, 2)),
):
    """Run MPPI planner locally (no server)."""
    from car_dynamics.controllers_jax import create_planner, ObstacleMap
    import jax.numpy as jnp

    if boxes is None:
        boxes = [
            ([0.5, -0.3], [1.2, 0.3]),
            ([-0.3, 0.5], [0.3, 1.2]),
        ]

    obs_map = ObstacleMap.from_boxes(boxes, bounds=bounds, resolution=0.05, inflate_radius=0.15)
    planner, _ = create_planner(
        dynamics="robocasa_holonomic",
        obstacle_map=obs_map,
        dt=1.0,
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
        dt_fwd=0.06,
        dt_side=0.06,
        dt_yaw=0.66,
    )

    state = jnp.array(start)
    goal_jnp = jnp.array(goal)
    goal_list = jnp.tile(goal_jnp, (planner.H + 1, 1))
    rp = planner.get_init_params()

    trajectory = [np.array(state)]
    actions = []
    DT_FWD = 0.06
    DT_SIDE = 0.06
    DT_YAW = 0.66

    print(f"Planning from {start} to {goal}...")
    t0 = time.time()
    for step in range(150):
        action, rp, info = planner(state, goal_list, rp, ())
        rp = planner.feed_hist(rp, state, action)
        actions.append(np.array(action))

        vf, vs, omega = float(action[0]), float(action[1]), float(action[2])
        theta = float(state[2])
        dx_b = vf * DT_FWD
        dy_b = vs * DT_SIDE
        x = float(state[0]) + (dx_b * np.cos(theta) - dy_b * np.sin(theta))
        y = float(state[1]) + (dx_b * np.sin(theta) + dy_b * np.cos(theta))
        state = jnp.array([x, y, theta + omega * DT_YAW])
        trajectory.append(np.array(state))

        pos_err = np.linalg.norm(np.array([x, y]) - np.array(goal[:2]))
        if pos_err < 0.05:
            print(f"  Reached goal in {step + 1} steps ({time.time() - t0:.2f}s)")
            break

    return {
        "trajectory": np.stack(trajectory),
        "actions": np.stack(actions),
        "sdf": np.array(obs_map.sdf),
        "sdf_origin": np.array(obs_map.origin),
        "sdf_resolution": obs_map.resolution,
        "boxes": boxes,
        "bounds": bounds,
        "start": start,
        "goal": goal,
    }


def plan_via_server(host, port, start, goal, env_port=None):
    """Plan via the remote MPPI server (optionally fetch collision geoms from env server)."""
    import portal

    client = portal.Client(f"{host}:{port}", logging=False)
    assert client.health_check().result(timeout=5), "MPPI server not reachable"

    if env_port:
        env_client = portal.Client(f"{host}:{env_port}", logging=False)
        coll = env_client.get_collision_geoms(2.0).result(timeout=10)
        positions = np.asarray(coll["positions"])
        dims = np.asarray(coll["dims_array"])
        bp = np.asarray(coll["base_pos"])

        if start is None:
            from scipy.spatial.transform import Rotation as R
            bq = np.asarray(coll["base_quat_xyzw"])
            yaw = float(R.from_quat(bq).as_euler("xyz")[2])
            start = (float(bp[0]), float(bp[1]), yaw)

        # Build boxes for viz
        boxes_raw = []
        for i in range(int(coll["n_geoms"])):
            z = float(positions[i, 2])
            h = float(dims[i, 2])
            if z - h / 2 > 0.8 or z + h / 2 < 0.05:
                continue
            cx, cy = float(positions[i, 0]), float(positions[i, 1])
            hw, hh = float(dims[i, 0]) / 2, float(dims[i, 1]) / 2
            boxes_raw.append([cx, cy, hw, hh])

        half = dims[:, :2] / 2
        floor_min = np.min(positions[:, :2] - half, axis=0) - 0.3
        floor_max = np.max(positions[:, :2] + half, axis=0) + 0.3
        floor_min = np.maximum(floor_min, bp[:2] - 3.0)
        floor_max = np.minimum(floor_max, bp[:2] + 3.0)

        client.update_world({
            "boxes": boxes_raw,
            "bounds_min": floor_min.tolist(),
            "bounds_max": floor_max.tolist(),
            "resolution": 0.08,
            "inflate_radius": 0.15,
        }).result(timeout=120)
        boxes_viz = [([b[0] - b[2], b[1] - b[3]], [b[0] + b[2], b[1] + b[3]]) for b in boxes_raw]
        bounds = (floor_min, floor_max)
    else:
        # Use dummy obstacles
        boxes_raw = [[0.5, 0.0, 0.3, 0.3], [-0.5, -0.5, 0.2, 0.2]]
        client.update_world({
            "boxes": boxes_raw,
            "bounds_min": [-2.0, -2.0],
            "bounds_max": [3.0, 2.0],
            "resolution": 0.05,
            "inflate_radius": 0.15,
        }).result(timeout=120)
        boxes_viz = [([b[0] - b[2], b[1] - b[3]], [b[0] + b[2], b[1] + b[3]]) for b in boxes_raw]
        bounds = ([-2, -2], [3, 2])

    result = client.plan_trajectory({
        "current_pos": list(start[:2]),
        "current_yaw": float(start[2]),
        "target_pos": list(goal[:2]),
        "target_yaw": float(goal[2]),
        "max_steps": 150,
    }).result(timeout=120)

    print(f"Plan: {result['status']}, {result['n_steps']} steps, "
          f"pos_err={result['final_pos_error']:.3f}m, yaw_err={result['final_yaw_error']:.3f}rad")

    return {
        "trajectory": np.asarray(result["trajectory"]),
        "actions": np.asarray(result["actions"]),
        "boxes": boxes_viz,
        "bounds": bounds,
        "start": start,
        "goal": goal,
    }


def visualize(data):
    """Plot SDF (if available), obstacles, trajectory, and actions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    traj = data["trajectory"]
    actions = data["actions"]
    boxes = data["boxes"]
    bounds = data["bounds"]
    start = data["start"]
    goal = data["goal"]

    # --- Panel 1: Obstacle map + trajectory ---
    ax = axes[0]
    ax.set_title("Trajectory + Obstacles")
    ax.set_aspect("equal")

    if "sdf" in data:
        sdf = data["sdf"]
        origin = data["sdf_origin"]
        res = data["sdf_resolution"]
        extent = [origin[0], origin[0] + sdf.shape[0] * res,
                  origin[1], origin[1] + sdf.shape[1] * res]
        ax.imshow(sdf.T, origin="lower", extent=extent, cmap="RdYlGn", alpha=0.5,
                  vmin=-0.3, vmax=0.5)

    for box in boxes:
        bmin, bmax = np.array(box[0]), np.array(box[1])
        w, h = bmax - bmin
        rect = patches.Rectangle(bmin, w, h, linewidth=1, edgecolor="red",
                                 facecolor="red", alpha=0.3)
        ax.add_patch(rect)

    ax.plot(traj[:, 0], traj[:, 1], "b.-", markersize=3, linewidth=1.5, label="trajectory")
    ax.plot(start[0], start[1], "go", markersize=10, label="start")
    ax.plot(goal[0], goal[1], "r*", markersize=15, label="goal")

    # Draw heading arrows every 10 steps
    for i in range(0, len(traj), max(1, len(traj) // 15)):
        dx = 0.05 * np.cos(traj[i, 2])
        dy = 0.05 * np.sin(traj[i, 2])
        ax.arrow(traj[i, 0], traj[i, 1], dx, dy, head_width=0.02, color="blue", alpha=0.5)

    ax.legend(fontsize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # --- Panel 2: Actions over time ---
    ax = axes[1]
    ax.set_title("Actions over time")
    t = np.arange(len(actions))  # step index
    ax.plot(t, actions[:, 0], label="v_fwd", linewidth=1.5)
    ax.plot(t, actions[:, 1], label="v_side", linewidth=1.5)
    ax.plot(t, actions[:, 2], label="omega", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Action value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Position error over time ---
    ax = axes[2]
    ax.set_title("Distance to goal")
    goal_xy = np.array(goal[:2])
    dists = np.linalg.norm(traj[:, :2] - goal_xy, axis=1)
    t_traj = np.arange(len(traj))  # step index
    ax.plot(t_traj, dists, "b-", linewidth=1.5)
    ax.axhline(y=0.05, color="g", linestyle="--", alpha=0.5, label="goal tol")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/mppi_debug.png", dpi=150)
    print("Saved: /tmp/mppi_debug.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Debug MPPI planner")
    parser.add_argument("--standalone", action="store_true", help="Run locally without server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18700, help="MPPI server port")
    parser.add_argument("--env-port", type=int, default=0, help="Env server port (0=skip, use dummy obstacles)")
    parser.add_argument("--start", nargs=3, type=float, default=None, help="Start x y yaw")
    parser.add_argument("--goal", nargs=3, type=float, default=[1.5, 0.5, 0.0], help="Goal x y yaw")
    args = parser.parse_args()

    start = tuple(args.start) if args.start else (0.0, 0.0, 0.0)
    goal = tuple(args.goal)

    if args.standalone:
        data = plan_standalone(start=start, goal=goal)
    else:
        data = plan_via_server(
            args.host, args.port,
            start=start if args.start else None,
            goal=goal,
            env_port=args.env_port if args.env_port > 0 else None,
        )

    visualize(data)


if __name__ == "__main__":
    main()
