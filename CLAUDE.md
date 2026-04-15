# anycar

General-purpose vehicle dynamics, planning, and control library built on JAX. Originally for RC car racing, now extended to support any 2D robot with collision-aware MPPI planning.

## Project Structure

```
anycar/
  car_dynamics/       # Core library (JAX)
    controllers_jax/  # MPPI controller, collision avoidance, dynamics presets
    models_jax/       # Dynamic bicycle model, neural dynamics (Transformer)
    models_torch/     # PyTorch dynamics models
    controllers_torch/# PyTorch controllers
    envs/             # Sim environments (MuJoCo, Isaac Sim, Assetto Corsa)
    modules/          # Safety modules (CBF)
  car_foundation/     # Model training (Transformer dynamics, dataset processing)
  car_planner/        # Trajectory planning (global trajectory, Frenet frames)
  car_collect/        # Data collection scripts (Assetto Corsa, MuJoCo, Isaac Sim)
  car_dataset/        # Dataset utilities
  car_ros2/           # ROS2 deployment nodes
```

## Key Modules

### MPPI Controller (`car_dynamics/controllers_jax/`)

The core planning algorithm. Two usage modes:

**1. Car-specific (original):** `MPPIController` with DBM or Transformer dynamics
- `mppi.py` — Base MPPI algorithm (sampling, rollout, weighting, update)
- `mppi_helper.py` — Car-specific reward/rollout factories
- Config: 600 rollouts, 50-step horizon, spline-based action sampling

**2. General-purpose (new):** `create_planner()` factory with any dynamics + collision avoidance
- `planner.py` — `create_planner()` factory, `CollisionMPPI` subclass
- `collision.py` — `ObstacleMap`: converts any obstacle format to SDF
- `dynamics_presets.py` — Holonomic, unicycle, differential drive models

### Dynamics Models (`car_dynamics/models_jax/`)

- `dbm.py` — Dynamic Bicycle Model (RK4 integration, 18 tire/vehicle params)
- `nn_dynamics.py` — JAX Transformer decoder for learned dynamics
- State: `[x, y, psi, vx, vy, omega]` (6D)
- Action: `[throttle, steering]` (2D)

### Safety (`car_dynamics/modules/`)

- `cbf.py` — Control Barrier Functions (CasADi QP, separate from MPPI)

### Trajectory Planning (`car_planner/`)

- `global_trajectory.py` — B-spline reference trajectories, Frenet coordinates, KD-tree queries

## General-Purpose MPPI API

```python
from car_dynamics.controllers_jax import create_planner, ObstacleMap

# Create obstacle map from any format
obstacle_map = ObstacleMap.from_boxes(boxes, bounds)       # bounding boxes
obstacle_map = ObstacleMap.from_occupancy_grid(grid, ...)  # binary grid
obstacle_map = ObstacleMap.from_point_cloud(points, ...)   # lidar/depth
obstacle_map = ObstacleMap.from_mesh_2d(verts, faces, ...) # 3D mesh
obstacle_map = ObstacleMap.from_circles(centers, radii, ...)

# Create planner (dynamics: 'holonomic', 'unicycle', 'diff_drive', or callable)
planner, running_params = create_planner(
    dynamics='holonomic',
    obstacle_map=obstacle_map,
    n_rollouts=200,
    dt=0.05,
)

# Control loop
action, running_params, info = planner(state, goal_trajectory, running_params, ())
```

## Running

```bash
# Install with uv
uv sync

# Run any script
uv run python <script.py>

# Run tests
uv run pytest
```

## Conventions

- Python 3.10+, managed with `uv`
- JAX for GPU-accelerated planning/dynamics (JIT + vmap)
- PyTorch for model training (car_foundation)
- NumPy arrays at boundaries, JAX arrays internally
