# General-Purpose MPPI Planner

Extends anycar's MPPI controller to work with any 2D robot and obstacle format.

## Motivation

The original MPPI (`mppi.py`) was car-specific: hardcoded 2-action dimensions, car-only reward function, no collision avoidance. Downstream projects (RoboCasa kitchen navigation, home-robot obstacle avoidance) need a general planner that accepts arbitrary obstacle inputs and robot dynamics.

## Architecture

Two-layer planner: global kinodynamic RRT for long-range feasible paths, local MPPI for smooth reactive tracking.

```
User code
  |
  +-- ObstacleMap.from_*(...)               # Any obstacle format -> SDF
  |     collision.py
  |
  +-- plan_global_path(dynamics, ...)       # [NEW] Kinodynamic RRT global path
  |     global_planner.py
  |        |
  |        +-- KinodynamicRRT              # RRT with dynamics propagation
  |        +-- GLOBAL_DYNAMICS_REGISTRY    # Numpy step fns (same models as MPPI)
  |
  +-- track_path_goal_list(path, state, H) # [NEW] Sliding-window waypoints
  |     global_planner.py
  |
  +-- create_planner(dynamics=..., obstacle_map=...)
  |     planner.py                          # Factory -> CollisionMPPI
  |        |
  |        +-- dynamics_presets.py          # holonomic / unicycle / diff_drive
  |        +-- CollisionMPPI               # MPPIController + SDF collision cost
  |              +-- mppi.py               # Core MPPI algorithm (unchanged)
  |
  +-- planner(state, goal_list, params, ()) # Control loop call
```

### Why two layers?

MPPI is a local optimizer with a fixed horizon (~4m reach at default settings). For targets beyond this range, or behind obstacles, a single-goal MPPI degrades: it can't "see" the goal, gets stuck in local minima, or greedily heads toward the goal and hits obstacles.

The global planner (kinodynamic RRT) solves this by finding a coarse collision-free path that respects the robot's dynamics. MPPI then tracks this path locally, providing smooth actions and reactive collision avoidance.

```
Far target (>2m):  RRT global path  ->  sliding-window goal_list  ->  MPPI
Close target:      Single tiled goal  ->  MPPI (unchanged)
```

## New Modules

### `global_planner.py` -- Kinodynamic RRT + Path Utilities

#### `KinodynamicRRT`

Grows a tree of dynamically feasible states from start toward goal:

1. **Sample** a random target (30% goal-biased)
2. **Find nearest** node in tree (brute-force on positions, fast for <50K nodes)
3. **Try N random actions**, propagate through dynamics for `extend_steps`
4. **Collision-check** each step via SDF lookup
5. **Add best** (closest to target, collision-free) node to tree
6. **Check goal** — if within position + heading tolerance, extract path

Key design choices:
- **Pure numpy dynamics** — no JAX overhead in the inner loop. The `make_*_step()` factories mirror `dynamics_presets.py` exactly, so global and local planners use the same dynamics.
- **Pre-allocated arrays** — no per-iteration object allocation.
- **Brute-force nearest neighbor** — faster than KDTree for <50K 2D nodes.

#### `plan_global_path()` -- Convenience API

One-call function matching `create_planner()` conventions:

```python
from car_dynamics.controllers_jax import plan_global_path

path = plan_global_path(
    start=np.array([0, 0, 0]),
    goal=np.array([3, 2, 1.57]),
    dynamics='holonomic',        # same string as create_planner()
    obstacle_map=obs_map,
)
# path: list of (3,) numpy arrays, or None
```

#### `track_path_goal_list()` -- Sliding Window

Extracts (H+1, D) goal_list for MPPI from the current position along the global path:

```python
from car_dynamics.controllers_jax import track_path_goal_list

# In control loop:
goal_list = track_path_goal_list(path, current_state, horizon=73)
action, rp, info = planner(state, jnp.array(goal_list), rp, ())
```

- Finds closest point on path to current state
- Linearly interpolates remaining path to H+1 waypoints
- Handles heading wraparound (unwrap before interpolation)
- As robot progresses, the window slides forward automatically

#### Dynamics Step Functions

Numpy mirrors of `dynamics_presets.py` for use in RRT:

| Factory | State | Action | MPPI equivalent |
|---------|-------|--------|-----------------|
| `make_holonomic_step(dt)` | [x, y, th] | [vf, vs, w] | `holonomic_rollout_fn` |
| `make_robocasa_holonomic_step(dt_fwd, dt_side, dt_yaw)` | [x, y, th] | [vf, vs, w] | `robocasa_holonomic_rollout_fn` |
| `make_unicycle_step(dt, wheelbase)` | [x, y, th, v] | [accel, steer] | `unicycle_rollout_fn` |
| `make_diff_drive_step(dt, wheel_sep)` | [x, y, th] | [vl, vr] | `diff_drive_rollout_fn` |

### `collision.py` -- ObstacleMap

Converts any obstacle representation to a Signed Distance Field (SDF) grid.

**Why SDF?** MPPI needs smooth cost gradients to distinguish "barely free" from "far from obstacles." Binary occupancy gives step-function costs that confuse the trajectory weighting. SDF gives smooth exponential falloff naturally. The RRT also uses the SDF for fast collision checking.

**Constructors:**
| Method | Input | Use case |
|--------|-------|----------|
| `from_occupancy_grid` | Binary grid + origin + resolution | home-robot voxel maps |
| `from_boxes` | List of (min_xy, max_xy) AABBs | RoboCasa fixture bounding boxes |
| `from_point_cloud` | (N, 2) points | Lidar scans |
| `from_mesh_2d` | Vertices + faces + height filter | MuJoCo scene meshes |
| `from_circles` | Centers + radii | Simple obstacle lists |

All constructors produce an `ObstacleMap` with `.sdf`, `.origin`, `.resolution`.

**Query:** `ObstacleMap.query_sdf_batch(positions, sdf, origin, resolution)` -- JIT-compiled batch lookup, called inside MPPI cost function.

### `dynamics_presets.py` -- Robot Dynamics (JAX)

Lightweight kinematic models matching the MPPI rollout interface.

| Preset | State | Action | Use case |
|--------|-------|--------|----------|
| `holonomic` | [x, y, th] | [v_fwd, v_side, w] | RoboCasa PandaOmron |
| `unicycle` | [x, y, th, v] | [accel, steer] | Simple car, Ackermann |
| `diff_drive` | [x, y, th] | [v_left, v_right] | TurtleBot, Stretch |

Custom dynamics: pass any `rollout_fn(obs_history, state, action, params, debug) -> (next_state, {})`.

### `planner.py` -- Factory API

`create_planner()` composes dynamics + obstacles + MPPI into a ready-to-use controller.

`CollisionMPPI` extends `MPPIController` with a composable reward:
- Goal position tracking (quadratic)
- Goal heading tracking (quadratic, wrapped angle)
- SDF collision penalty (exponential proximity + hard inside penalty)
- Action smoothness (quadratic delta)

## Full Usage Example

```python
import numpy as np
import jax.numpy as jnp
from car_dynamics.controllers_jax import (
    create_planner, ObstacleMap, plan_global_path, track_path_goal_list,
)

# 1. Build obstacle map
obs_map = ObstacleMap.from_boxes(
    [([0.5, -0.3], [1.2, 0.3])],
    bounds=([-2, -2], [3, 2]),
)

# 2. Create local MPPI planner
planner, rp = create_planner(
    dynamics='holonomic', obstacle_map=obs_map,
    n_rollouts=2048, dt=0.05,
)

# 3. Plan global path (for far targets)
start = np.array([0.0, 0.0, 0.0])
goal = np.array([2.5, 1.5, 1.57])

global_path = plan_global_path(
    start, goal, dynamics='holonomic', obstacle_map=obs_map, dt=0.05,
)

# 4. Control loop: MPPI tracks global path via sliding window
state = jnp.array(start)
for step in range(200):
    if global_path is not None:
        goal_list = jnp.array(
            track_path_goal_list(global_path, state, planner.H)
        )
    else:
        goal_list = jnp.tile(jnp.array(goal), (planner.H + 1, 1))

    action, rp, info = planner(state, goal_list, rp, ())
    rp = planner.feed_hist(rp, state, action)
    state = step_dynamics(state, action)  # your dynamics
```

## Server Integration

`serve_mppi_planner.py` automatically uses the global planner when the target is beyond MPPI's local reach (50% of horizon distance). The decision is transparent:

```
Client: plan_trajectory({current_pos, target_pos, ...})
Server:
  1. dist = ||current - target||
  2. if dist > horizon_reach * 0.5:
       global_path = plan_global_path(...)  # ~10-100ms
  3. for step in range(max_steps):
       goal_list = track_path_goal_list(global_path, state, H)  # ~0.01ms
       action = mppi(state, goal_list, ...)                      # ~5-10ms
  4. return trajectory, actions, status, used_global_planner
```

Response includes `used_global_planner: bool` so the client knows which layer was active.

## Changes to Existing Code

### `mppi.py` (4 edits, all backward-compatible)

1. **Line 68**: Added `'scan'` dynamics type -> reuses `lax.scan` path (same as `'dbm'`)
2. **Line 82**: `[sigma] * 2` -> `[sigma] * num_actions` -- fixes hardcoded 2-action assumption
3. **Lines 379-381**: Spline interpolation loop over `num_actions` instead of hardcoded indices 0, 1
4. **Lines 391-393**: Same for sampled action interpolation
5. **Line 9**: Removed unused `DynamicParams` import (broke import chain via `transformer_engine`)

### `models_jax/__init__.py` (1 edit)

Made `DynamicsJax` import lazy via `__getattr__` to avoid pulling in `transformer_engine` (GPU-only NVIDIA package) when only using the controller modules.

### `serve_mppi_planner.py` (updated)

Added automatic global planner integration in `_handle_plan_trajectory`:
- Distance check to decide local-only vs global+local
- RRT path generation for far targets
- Sliding-window goal_list for MPPI tracking
- `used_global_planner` field in response

## Performance

### Local MPPI (close targets)

| Config | Value |
|--------|-------|
| Dynamics | Holonomic (3 multiplies per step) |
| Rollouts | 200 |
| Horizon | 10 steps (0.5s at 20Hz) |
| SDF grid | 80x80 at 5cm resolution |
| **Latency (CPU)** | **~30ms** |
| **Latency (GPU)** | **~5ms** |

### Global RRT (far targets)

| Config | Value |
|--------|-------|
| Grid | 100x100 SDF |
| Max iterations | 5000 |
| Extend steps | 5 |
| Action samples | 5 |
| **Typical time** | **10-100ms** (depends on distance + obstacle complexity) |
| **Then per-step MPPI** | **~5-10ms** (same as local) |
