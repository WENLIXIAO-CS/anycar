"""
High-level factory for creating collision-aware MPPI planners.

This is the main user-facing API. Downstream projects create a planner in ~3 lines:

    from car_dynamics.controllers_jax import create_planner, ObstacleMap

    obstacle_map = ObstacleMap.from_boxes(boxes, bounds)
    planner, running_params = create_planner(
        dynamics='holonomic',
        obstacle_map=obstacle_map,
    )

    # In control loop:
    action, running_params, info = planner(state, goal_trajectory, running_params, ())
"""

import jax
import jax.numpy as jnp
from functools import partial

from .mppi import MPPIController, MPPIParams
from .mppi_torch_style import MPPITorch, MPPITorchConfig
from .collision import ObstacleMap
from .dynamics_presets import DYNAMICS_REGISTRY
from .utils import void_fn


class CollisionMPPI(MPPIController):
    """
    MPPI controller with configurable collision cost via ObstacleMap.

    Extends the base MPPIController by overriding single_step_reward with
    a composable reward that includes:
        - Goal position tracking
        - Goal heading tracking
        - SDF-based collision penalty
        - Action smoothness penalty

    The obstacle_map is baked into the JIT-compiled reward via self (static_argnums=(0,)).
    To change the obstacle map (e.g., at episode reset), create a new CollisionMPPI instance.
    """

    def __init__(
        self,
        params,
        rollout_fn,
        key,
        obstacle_map=None,
        collision_weight=100.0,
        collision_decay=5.0,
        collision_inside_weight=1000.0,
        goal_pos_weight=10.0,
        goal_heading_weight=2.0,
        goal_vel_weight=0.0,
        smoothness_weight=0.5,
        terminal_weight=10.0,
        control_reg_weight=0.0,
        sign_change_weight=0.0,
        goal_reach_reward=0.0,
        goal_reach_threshold=0.1,
    ):
        """
        Args:
            params: MPPIParams configuration.
            rollout_fn: Dynamics rollout function.
            key: JAX PRNG key.
            obstacle_map: ObstacleMap instance (or None for no collision cost).
            collision_weight: Weight for exponential proximity penalty.
            collision_decay: Decay rate for proximity penalty (higher = sharper).
            collision_inside_weight: Hard penalty for being inside obstacles.
            goal_pos_weight: Weight for position tracking cost.
            goal_heading_weight: Weight for heading tracking cost.
            goal_vel_weight: Weight for velocity tracking cost (requires goal dim >= 4 and state dim >= 4).
            smoothness_weight: Weight for action smoothness cost.
            goal_reach_reward: Sparse bonus for rollouts where any state reaches the goal.
            goal_reach_threshold: Distance threshold for goal-reached check (meters).
        """
        super().__init__(params, rollout_fn, void_fn, key)

        # Store obstacle data as JAX arrays (baked into JIT trace)
        if obstacle_map is not None:
            self._has_obstacles = True
            self._sdf_grid = obstacle_map.sdf
            self._sdf_origin = obstacle_map.origin
            self._sdf_resolution = obstacle_map.resolution
        else:
            self._has_obstacles = False
            self._sdf_grid = jnp.zeros((1, 1))
            self._sdf_origin = jnp.zeros(2)
            self._sdf_resolution = 1.0

        self._cw = collision_weight
        self._cd = collision_decay
        self._ciw = collision_inside_weight
        self._gw_pos = goal_pos_weight
        self._gw_head = goal_heading_weight
        self._gw_vel = goal_vel_weight
        self._sw = smoothness_weight
        self._tw = terminal_weight
        self._crw = control_reg_weight
        self._scw = sign_change_weight
        self._grr = goal_reach_reward
        self._grt = goal_reach_threshold

    @partial(jax.jit, static_argnums=(0,))
    def single_step_reward(self, carry, pair):
        """
        Standard MPPI cost: running cost + terminal cost + control regularization.

        Running: q(x) = position + heading + velocity tracking + collision avoidance
        Terminal: phi(x_T) = terminal_weight * running cost at last step
        Control: lambda * u^T u (penalizes control effort)
        Smoothness: delta-u penalty (penalizes action changes)
        """
        step, prev_action = carry
        state_step, action_step, goal = pair

        # --- Running cost: goal tracking ---
        dist_pos = jnp.linalg.norm(state_step[:, :2] - goal[:2], axis=1)
        reward = -dist_pos ** 2 * self._gw_pos

        diff_psi = state_step[:, 2] - goal[2]
        diff_psi = jnp.arctan2(jnp.sin(diff_psi), jnp.cos(diff_psi))
        reward += -diff_psi ** 2 * self._gw_head

        if self._gw_vel > 0 and self.params.num_obs >= 4:
            diff_vel = state_step[:, 3] - goal[3]
            reward += -diff_vel ** 2 * self._gw_vel

        # --- Collision ---
        if self._has_obstacles:
            dist_obs = ObstacleMap.query_sdf_batch(
                state_step[:, :2],
                self._sdf_grid,
                self._sdf_origin,
                self._sdf_resolution,
            )
            reward += -self._cw * jnp.exp(-self._cd * jnp.clip(dist_obs, 0.0, None))
            reward += jnp.where(dist_obs < 0.0, -self._ciw, 0.0)

        # --- Terminal cost: extra weight at last horizon step ---
        # step is 0-indexed; H-1 is the last step. We use the total horizon
        # from the scan length. Since goal_list[1:] is passed, last step = H-1.
        # Apply terminal multiplier that increases toward the end.
        # At step t: weight = 1 + terminal_weight * (t / (H-1))^2
        H_approx = float(self.H - 1) if self.H > 1 else 1.0
        terminal_scale = 1.0 + self._tw * (step / H_approx) ** 2
        reward = reward * terminal_scale

        # --- Action smoothness: delta-u penalty ---
        reward += -self._sw * jnp.sum((action_step - prev_action) ** 2, axis=1)

        # --- Sign-change penalty: penalize when action flips direction ---
        # sign(a_t) != sign(a_{t-1}) → high cost
        sign_change = (action_step * prev_action) < 0  # True where signs differ
        reward += -self._scw * jnp.sum(sign_change.astype(jnp.float32), axis=1)

        # --- Control regularization: u^T u ---
        if self._crw > 0:
            reward += -self._crw * jnp.sum(action_step ** 2, axis=1)

        reward *= self.params.discount ** step
        return (step + 1, action_step), reward


class CollisionMPPITorch(MPPITorch):
    """
    MPPITorch with the same collision-aware cost function as CollisionMPPI.

    Uses the mppi_torch algorithm (per-timestep noise, adaptive covariance,
    adaptive temperature) instead of the spline-based original.
    """

    def __init__(
        self,
        config,
        rollout_fn,
        key,
        obstacle_map=None,
        collision_weight=100.0,
        collision_decay=5.0,
        collision_inside_weight=1000.0,
        goal_pos_weight=10.0,
        goal_heading_weight=2.0,
        goal_vel_weight=0.0,
        smoothness_weight=0.5,
        terminal_weight=10.0,
        control_reg_weight=0.0,
        sign_change_weight=0.0,
        goal_reach_reward=0.0,
        goal_reach_threshold=0.1,
    ):
        # Store cost params (baked into JIT via static_argnums=(0,))
        if obstacle_map is not None:
            self._has_obstacles = True
            self._sdf_grid = obstacle_map.sdf
            self._sdf_origin = obstacle_map.origin
            self._sdf_resolution = obstacle_map.resolution
        else:
            self._has_obstacles = False
            self._sdf_grid = jnp.zeros((1, 1))
            self._sdf_origin = jnp.zeros(2)
            self._sdf_resolution = 1.0

        self._cw = collision_weight
        self._cd = collision_decay
        self._ciw = collision_inside_weight
        self._gw_pos = goal_pos_weight
        self._gw_head = goal_heading_weight
        self._gw_vel = goal_vel_weight
        self._sw = smoothness_weight
        self._tw = terminal_weight
        self._crw = control_reg_weight
        self._scw = sign_change_weight
        self._grr = goal_reach_reward
        self._grt = goal_reach_threshold

        # Build cost_fn matching the scan interface
        super().__init__(config, rollout_fn, self._cost_step, key)

    @partial(jax.jit, static_argnums=(0,))
    def _cost_step(self, carry, pair):
        """Same cost function as CollisionMPPI.single_step_reward."""
        step, prev_action = carry
        state_step, action_step, goal = pair

        dist_pos = jnp.linalg.norm(state_step[:, :2] - goal[:2], axis=1)
        reward = -dist_pos ** 2 * self._gw_pos

        diff_psi = state_step[:, 2] - goal[2]
        diff_psi = jnp.arctan2(jnp.sin(diff_psi), jnp.cos(diff_psi))
        reward += -diff_psi ** 2 * self._gw_head

        if self._gw_vel > 0 and self.config.num_obs >= 4:
            diff_vel = state_step[:, 3] - goal[3]
            reward += -diff_vel ** 2 * self._gw_vel

        if self._has_obstacles:
            dist_obs = ObstacleMap.query_sdf_batch(
                state_step[:, :2],
                self._sdf_grid,
                self._sdf_origin,
                self._sdf_resolution,
            )
            reward += -self._cw * jnp.exp(-self._cd * jnp.clip(dist_obs, 0.0, None))
            reward += jnp.where(dist_obs < 0.0, -self._ciw, 0.0)

        # Terminal cost: increasing weight toward end of horizon
        H_approx = float(self.config.horizon - 1) if self.config.horizon > 1 else 1.0
        terminal_scale = 1.0 + self._tw * (step / H_approx) ** 2
        reward = reward * terminal_scale

        reward += -self._sw * jnp.sum((action_step - prev_action) ** 2, axis=1)

        sign_change = (action_step * prev_action) < 0
        reward += -self._scw * jnp.sum(sign_change.astype(jnp.float32), axis=1)

        if self._crw > 0:
            reward += -self._crw * jnp.sum(action_step ** 2, axis=1)

        reward *= self.config.discount ** step
        return (step + 1, action_step), reward


def create_planner(
    dynamics='holonomic',
    obstacle_map=None,
    dt=0.05,
    n_rollouts=200,
    horizon_knots=4,
    num_intermediate=3,
    sigma=0.3,
    lam=0.05,
    collision_weight=100.0,
    collision_decay=5.0,
    collision_inside_weight=1000.0,
    goal_pos_weight=10.0,
    goal_heading_weight=2.0,
    goal_vel_weight=0.0,
    smoothness_weight=0.5,
    terminal_weight=10.0,
    control_reg_weight=0.0,
    sign_change_weight=0.0,
    action_limit=1.0,
    goal_reach_reward=0.0,
    goal_reach_threshold=0.1,
    gamma_sigma=0.0,
    cov_floor=0.0,
    adaptive_beta=False,
    backend='spline',
    seed=42,
    **dynamics_kwargs,
):
    """
    Create a ready-to-use collision-aware MPPI planner.

    Args:
        dynamics: One of 'holonomic', 'unicycle', 'diff_drive', or a callable rollout_fn.
            If callable, must also pass num_obs and num_actions in dynamics_kwargs.
        obstacle_map: ObstacleMap instance, or None for no collision cost.
        dt: Control timestep in seconds.
        n_rollouts: Number of parallel trajectory samples.
        horizon_knots: Number of spline knot points.
        num_intermediate: Interpolation points between knots.
            Total horizon = (horizon_knots - 1) * num_intermediate + 1.
        sigma: Exploration noise standard deviation.
        lam: MPPI temperature (lower = more greedy).
        collision_weight: Weight for exponential proximity penalty.
        collision_decay: Decay rate for proximity penalty.
        collision_inside_weight: Hard penalty for being inside obstacles.
        goal_pos_weight: Weight for position tracking cost.
        goal_heading_weight: Weight for heading tracking cost.
        smoothness_weight: Weight for action smoothness cost.
        seed: Random seed.
        **dynamics_kwargs: Extra args passed to dynamics factory (e.g., wheelbase=0.3).

    Returns:
        (planner, running_params): planner is callable as
            action, running_params, info = planner(obs, goal_list, running_params, ())
    """
    # Resolve dynamics
    if isinstance(dynamics, str):
        if dynamics not in DYNAMICS_REGISTRY:
            raise ValueError(
                f"Unknown dynamics '{dynamics}'. "
                f"Available: {list(DYNAMICS_REGISTRY.keys())}. "
                f"Or pass a callable rollout_fn."
            )
        entry = DYNAMICS_REGISTRY[dynamics]
        rollout_fn = entry['factory'](dt, **dynamics_kwargs)
        num_obs = entry['num_obs']
        num_actions = entry['num_actions']
    elif callable(dynamics):
        rollout_fn = dynamics
        num_obs = dynamics_kwargs.get('num_obs')
        num_actions = dynamics_kwargs.get('num_actions')
        if num_obs is None or num_actions is None:
            raise ValueError(
                "When passing a custom rollout_fn, must also provide "
                "num_obs and num_actions in kwargs."
            )
    else:
        raise TypeError(f"dynamics must be str or callable, got {type(dynamics)}")

    params = MPPIParams(
        sigma=sigma,
        gamma_mean=1.0,
        gamma_sigma=gamma_sigma,
        discount=1.0,
        sample_sigma=1.0,
        lam=lam,
        n_rollouts=n_rollouts,
        h_knot=horizon_knots,
        a_min=-jnp.ones(num_actions) * action_limit,
        a_max=jnp.ones(num_actions) * action_limit,
        a_mag=jnp.ones(num_actions),
        a_shift=jnp.zeros(num_actions),
        delay=0,
        len_history=1,
        debug=False,
        fix_history=False,
        num_obs=num_obs,
        num_actions=num_actions,
        num_intermediate=num_intermediate,
        spline_order=2,
        dynamics='scan',
        cov_floor=cov_floor,
        adaptive_beta=adaptive_beta,
    )

    key = jax.random.PRNGKey(seed)

    if backend == 'torch':
        # mppi_torch-style: per-timestep noise, adaptive cov/temp
        horizon = (horizon_knots - 1) * num_intermediate + 1
        config = MPPITorchConfig(
            num_obs=num_obs,
            num_actions=num_actions,
            n_rollouts=n_rollouts,
            horizon=horizon,
            sigma=sigma,
            lam=lam,
            adaptive_beta=adaptive_beta,
            gamma_sigma=gamma_sigma,
            cov_floor=cov_floor,
            gamma_mean=1.0,
            a_min=-action_limit,
            a_max=action_limit,
        )
        planner = CollisionMPPITorch(
            config=config,
            rollout_fn=rollout_fn,
            key=key,
            obstacle_map=obstacle_map,
            collision_weight=collision_weight,
            collision_decay=collision_decay,
            collision_inside_weight=collision_inside_weight,
            goal_pos_weight=goal_pos_weight,
            goal_heading_weight=goal_heading_weight,
            goal_vel_weight=goal_vel_weight,
            smoothness_weight=smoothness_weight,
            terminal_weight=terminal_weight,
            control_reg_weight=control_reg_weight,
            sign_change_weight=sign_change_weight,
            goal_reach_reward=goal_reach_reward,
            goal_reach_threshold=goal_reach_threshold,
        )
    else:
        # Original spline-based MPPI
        planner = CollisionMPPI(
            params=params,
            rollout_fn=rollout_fn,
            key=key,
            obstacle_map=obstacle_map,
            collision_weight=collision_weight,
            collision_decay=collision_decay,
            collision_inside_weight=collision_inside_weight,
            goal_pos_weight=goal_pos_weight,
            goal_heading_weight=goal_heading_weight,
            goal_vel_weight=goal_vel_weight,
            smoothness_weight=smoothness_weight,
            terminal_weight=terminal_weight,
            control_reg_weight=control_reg_weight,
            sign_change_weight=sign_change_weight,
            goal_reach_reward=goal_reach_reward,
            goal_reach_threshold=goal_reach_threshold,
        )

    running_params = planner.get_init_params()
    return planner, running_params
