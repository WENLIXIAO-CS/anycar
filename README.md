
# AnyCar to Anywhere: Learning Universal Dynamics Model for Agile and Adaptive Mobility
<div align="center">

[[Website]](https://lecar-lab.github.io/anycar/)
[[Arxiv]](https://arxiv.org/abs/2409.15783)
[[Video]](https://www.youtube.com/)

[<img src="https://img.shields.io/badge/Backend-Jax-red.svg"/>](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<img src="./media/2024_AnyCar.gif" width="600px"/>

</div>

General-purpose vehicle dynamics, planning, and control library built on JAX. Originally developed for RC car racing (AnyCar), now extended to support any 2D robot with collision-aware MPPI planning.

## Project Structure

```
anycar/
  car_dynamics/              # Core library (JAX)
    controllers_jax/         # MPPI controller, collision avoidance, dynamics presets
      mppi.py                #   Base MPPI algorithm (sampling, rollout, weighting)
      planner.py             #   create_planner() factory, CollisionMPPI
      collision.py           #   ObstacleMap: any obstacle format -> SDF
      dynamics_presets.py    #   Holonomic, unicycle, differential drive models
      global_planner.py      #   Kinodynamic RRT global planner
      hybrid_astar.py        #   Hybrid A* planning
      mppi_torch_style.py    #   PyTorch-style MPPI variant
    models_jax/              # Dynamic bicycle model, neural dynamics (Transformer)
    models_torch/            # PyTorch dynamics models
    controllers_torch/       # PyTorch controllers
    envs/                    # Sim environments (MuJoCo, Isaac Sim, Assetto Corsa)
    modules/                 # Safety modules (CBF)
  car_foundation/            # Model training (Transformer dynamics, dataset processing)
  car_planner/               # Trajectory planning (global trajectory, Frenet frames)
  car_collect/               # Data collection scripts (Assetto Corsa, MuJoCo, Isaac Sim)
  car_dataset/               # Dataset utilities
  car_ros2/                  # ROS2 deployment nodes
  viz/                       # Web-based planner visualization (FastAPI + JS)
  docs/                      # Detailed documentation
  serve_mppi_planner.py      # Remote MPPI planner server (Portal RPC)
  debug_mppi_plan.py         # Standalone planner debugging script
```

## Setup

> [!IMPORTANT]
> We recommend Ubuntu >= 22.04 + Python >= 3.10 + CUDA >= 12.3.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)

2. Clone and install:

    ```bash
    git clone git@github.com:LeCAR-Lab/anycar.git
    cd anycar
    uv sync
    ```

3. Run any script:

    ```bash
    uv run python <script.py>
    ```

## Quick Start: General-Purpose MPPI Planner

The general-purpose planner works with any 2D robot dynamics and obstacle format. See [docs/GENERAL_MPPI_PLANNER.md](./docs/GENERAL_MPPI_PLANNER.md) for the full API reference.

```python
import numpy as np
import jax.numpy as jnp
from car_dynamics.controllers_jax import (
    create_planner, ObstacleMap, plan_global_path, track_path_goal_list,
)

# 1. Build obstacle map from any format
obs_map = ObstacleMap.from_boxes(
    [([0.5, -0.3], [1.2, 0.3])],
    bounds=([-2, -2], [3, 2]),
)

# 2. Create planner (dynamics: 'holonomic', 'unicycle', 'diff_drive', or callable)
planner, rp = create_planner(
    dynamics='holonomic', obstacle_map=obs_map,
    n_rollouts=2048, dt=0.05,
)

# 3. (Optional) Global path for far targets
start = np.array([0.0, 0.0, 0.0])
goal = np.array([2.5, 1.5, 1.57])
global_path = plan_global_path(
    start, goal, dynamics='holonomic', obstacle_map=obs_map, dt=0.05,
)

# 4. Control loop
state = jnp.array(start)
for step in range(200):
    if global_path is not None:
        goal_list = jnp.array(track_path_goal_list(global_path, state, planner.H))
    else:
        goal_list = jnp.tile(jnp.array(goal), (planner.H + 1, 1))

    action, rp, info = planner(state, goal_list, rp, ())
    rp = planner.feed_hist(rp, state, action)
    state = step_dynamics(state, action)  # your dynamics
```

**Obstacle map constructors:** `from_boxes`, `from_occupancy_grid`, `from_point_cloud`, `from_mesh_2d`, `from_circles`

**Built-in dynamics:** holonomic (3-state), unicycle (4-state), differential drive (3-state), or pass any custom rollout function.

### Server Mode

Run the planner as a remote service via Portal RPC:

```bash
uv run python serve_mppi_planner.py --port 18700
```

### Visualization

```bash
uv run python viz/app.py
```

## Quick Start: RC Car Simulation (ROS2)

> [!NOTE]
> ROS2 deployment requires additional setup. Install [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html) first.

1. Set up ROS2 Python compatibility:

    ```bash
    export PYTHON_EXECUTABLE=$(which python)
    export PYTHONPATH=$(python -c "import site; print(site.getsitepackages()[0])"):$PYTHONPATH
    export CAR_PATH=$(pwd)
    colcon build
    source install/setup.bash
    ```

2. Download [Foxglove Studio](https://foxglove.dev/download) and import config from `misc/anycar-vis.json`

3. Launch:

    ```bash
    ros2 launch foxglove_bridge foxglove_bridge_launch.xml
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 ros2 launch car_ros2 car_sim.launch.py
    ```

4. Open Foxglove Studio at `localhost:8765`

<div align="center">
<img src="./media/foxglove.gif" width="600px"/>
</div>

> For this example environment, we provide an [example checkpoint](https://huggingface.co/wenlixiao/anycar-sim-small/tree/main/anycar_model_checkpoint). Please put folder under `car_foundation/car_foundation/models/`.

## AnyCar Pipeline

- **Data Collection** — See [car_collect/README.md](./car_collect/README.md). Data saves to `car_foundation/car_foundation/data`.
- **Model Training** — See [car_foundation/README.md](./car_foundation/README.md). Models save to `car_foundation/car_foundation/models`.
- **Controller** — See [car_dynamics/README.md](./car_dynamics/README.md) for dynamics models and MPPI implementation.
- **Deployment** — See [car_ros2/README.md](./car_ros2/README.md) for ROS2 sim/real deployment.
- **Hardware** — See [hardware/README.md](./hardware/README.md) for car configurations and 3D models.

## Citation
```bibtex
@misc{xiao2024anycaranywherelearninguniversal,
      title={AnyCar to Anywhere: Learning Universal Dynamics Model for Agile and Adaptive Mobility}, 
      author={Wenli Xiao and Haoru Xue and Tony Tao and Dvij Kalaria and John M. Dolan and Guanya Shi},
      year={2024},
      eprint={2409.15783},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.15783}, 
}
```
