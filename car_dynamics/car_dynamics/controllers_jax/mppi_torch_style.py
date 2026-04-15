"""
MPPI controller matching the mppi_torch (tud-amr) algorithm.

Key differences from the spline-based mppi.py:
  - Per-timestep independent noise (no spline knots)
  - Adaptive diagonal covariance (shrinks near convergence, grows during exploration)
  - Adaptive temperature (auto-tunes based on effective sample count)
  - Optional Savitzky-Golay post-filtering

Reference: https://github.com/tud-amr/mppi_torch

Usage:
    from car_dynamics.controllers_jax.mppi_torch_style import MPPITorch

    mppi = MPPITorch(config, rollout_fn, cost_fn, key)
    running = mppi.get_init_params()
    action, running, info = mppi(state, goal_list, running, dyn_params)
"""

import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
import flax


# ---------------------------------------------------------------------------
# Config & Running State
# ---------------------------------------------------------------------------

@dataclass
class MPPITorchConfig:
    """Static configuration (baked into JIT trace via static_argnums)."""
    # Dimensions
    num_obs: int
    num_actions: int
    # Sampling
    n_rollouts: int = 2048
    horizon: int = 64
    sigma: object = 0.3         # float or list[float] per action dim
    # Temperature
    lam: float = 0.05           # initial temperature (beta)
    adaptive_beta: bool = True
    beta_eta_upper: float = 50.0
    beta_eta_lower: float = 5.0
    beta_shrink: float = 0.9
    beta_grow: float = 1.2
    beta_min: float = 0.001
    beta_max: float = 1.0
    # Covariance adaptation
    gamma_sigma: float = 0.5    # EMA rate for covariance update (0=fixed, 1=fully adaptive)
    cov_floor: float = 0.0001   # kappa: minimum covariance to prevent collapse
    # Mean update
    gamma_mean: float = 1.0     # 1.0 = full update (no momentum)
    # Action bounds
    a_min: object = -1.0        # float or array
    a_max: object = 1.0
    # Discount
    discount: float = 1.0


@flax.struct.dataclass
class MPPITorchRunning:
    """Mutable state that persists across MPPI calls."""
    u_mean: jnp.ndarray     # (H, nu) mean action sequence
    cov_diag: jnp.ndarray   # (nu,) diagonal covariance per action dim
    beta: float              # adaptive temperature
    key: jax.random.PRNGKey


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MPPITorch:
    """
    MPPI controller with per-timestep noise, adaptive covariance, and
    adaptive temperature. Matches the mppi_torch algorithm.
    """

    def __init__(self, config: MPPITorchConfig, rollout_fn, cost_fn, key):
        """
        Args:
            config: MPPITorchConfig with all hyperparameters.
            rollout_fn: JAX-compatible dynamics function.
                Signature: rollout_fn(obs_hist, state, action, dyn_params, debug)
                    → (next_state, {})
                Where state is (K, num_obs), action is (K, num_actions).
            cost_fn: Callable(carry, (state_step, action_step, goal)) → (carry, reward).
                Same signature as CollisionMPPI.single_step_reward.
            key: JAX PRNG key.
        """
        self.config = config
        self.rollout_fn = rollout_fn
        self.cost_fn = cost_fn
        self.H = config.horizon
        self.nu = config.num_actions
        self.K = config.n_rollouts

        # Parse sigma into per-action array
        s = config.sigma
        self._sigma_init = jnp.array(s) if hasattr(s, '__len__') else jnp.full(self.nu, s)

        # Parse action bounds
        self._a_min = jnp.array(config.a_min) if hasattr(config.a_min, '__len__') else jnp.full(self.nu, config.a_min)
        self._a_max = jnp.array(config.a_max) if hasattr(config.a_max, '__len__') else jnp.full(self.nu, config.a_max)

    @property
    def params(self):
        """Compatibility shim so CollisionMPPI can access config fields."""
        return self.config

    def get_init_params(self):
        return MPPITorchRunning(
            u_mean=jnp.zeros((self.H, self.nu)),
            cov_diag=self._sigma_init ** 2,
            beta=self.config.lam,
            key=jax.random.PRNGKey(123),
        )

    @partial(jax.jit, static_argnums=(0,))
    def feed_hist(self, running: MPPITorchRunning, obs, action):
        """No-op for compatibility — this controller doesn't use state history."""
        return running

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, obs, goal_list, running: MPPITorchRunning, dyn_params):
        """
        Run one MPPI iteration.

        Args:
            obs: (num_obs,) current state.
            goal_list: (H+1, goal_dims) goal trajectory.
            running: MPPITorchRunning state.
            dyn_params: tuple passed to rollout_fn.

        Returns:
            action: (nu,) best action to execute now.
            new_running: updated MPPITorchRunning.
            info: dict with 'trajectory', 'a_mean_jnp'.
        """
        K, H, nu = self.K, self.H, self.nu
        beta = running.beta

        # --- 1. Sample noise: (K, H, nu) ~ N(0, diag(cov_diag)) ---
        key, key_noise = jax.random.split(running.key)
        scale = jnp.sqrt(running.cov_diag)  # (nu,)
        noise = jax.random.normal(key_noise, (K, H, nu)) * scale[None, None, :]

        # --- 2. Perturbed actions = mean + noise, clamped ---
        perturbed = running.u_mean[None, :, :] + noise  # (K, H, nu)
        perturbed = jnp.clip(perturbed, self._a_min[None, None, :], self._a_max[None, None, :])

        # Actual noise after clipping (for correct covariance update)
        noise_clipped = perturbed - running.u_mean[None, :, :]

        # --- 3. Rollout all K trajectories ---
        state_init = jnp.tile(obs[None, :], (K, 1))  # (K, num_obs)

        def scan_step(carry, action_t):
            state = carry
            next_state, _ = self.rollout_fn(None, state, action_t, dyn_params, False)
            return next_state, next_state

        # perturbed: (K, H, nu) → need (H, K, nu) for scan
        actions_scan = jnp.swapaxes(perturbed, 0, 1)  # (H, K, nu)
        _, state_traj = jax.lax.scan(scan_step, state_init, actions_scan)
        # state_traj: (H, K, num_obs)
        # Prepend initial state
        state_traj = jnp.concatenate([state_init[None, :, :], state_traj], axis=0)
        # state_traj: (H+1, K, num_obs)

        # --- 4. Compute costs via cost_fn (same interface as CollisionMPPI) ---
        actions_for_cost = jnp.swapaxes(perturbed, 0, 1)  # (H, K, nu)
        _, reward_list = jax.lax.scan(
            self.cost_fn,
            (0, actions_for_cost[0]),
            (state_traj[1:], actions_for_cost, goal_list[1:]),
        )
        rewards = jnp.sum(reward_list, axis=0)  # (K,)

        # Sparse goal-reach bonus: position + heading + stopped
        if hasattr(self, '_grr') and self._grr > 0:
            goal_pos = goal_list[-1, :2]
            goal_heading = goal_list[-1, 2]
            dist_to_goal = jnp.linalg.norm(
                state_traj[:, :, :2] - goal_pos[None, None, :], axis=2
            )
            pos_ok = dist_to_goal < self._grt
            heading_diff = jnp.abs(jnp.arctan2(
                jnp.sin(state_traj[:, :, 2] - goal_heading),
                jnp.cos(state_traj[:, :, 2] - goal_heading),
            ))
            heading_ok = heading_diff < (self._grt * 5.0)
            if self.config.num_obs >= 4:
                vel_ok = jnp.abs(state_traj[:, :, 3]) < 0.3
            else:
                vel_ok = jnp.ones_like(pos_ok)
            reached_mask = pos_ok & heading_ok & vel_ok
            first_step = jnp.argmax(reached_mask, axis=0)
            ever_reached = jnp.any(reached_mask, axis=0).astype(jnp.float32)
            H_float = float(state_traj.shape[0] - 1)
            reward_bonus = ever_reached * self._grr * jnp.exp(-3.0 * first_step / H_float)
            rewards = rewards + reward_bonus

        costs = -rewards

        # --- 5. Compute weights with temperature ---
        cost_min = jnp.min(costs)
        cost_exp = jnp.exp(-(costs - cost_min) / beta)
        eta = jnp.sum(cost_exp)
        weights = cost_exp / eta  # (K,)

        # --- 6. Adaptive temperature ---
        new_beta = jnp.where(
            self.config.adaptive_beta,
            jnp.clip(
                jnp.where(eta > self.config.beta_eta_upper, beta * self.config.beta_shrink,
                jnp.where(eta < self.config.beta_eta_lower, beta * self.config.beta_grow,
                beta)),
                self.config.beta_min, self.config.beta_max,
            ),
            beta,
        )

        # --- 7. Update mean: weighted sum of noise added to current mean ---
        # u_new = u_old + sum(w_i * noise_i)
        weighted_noise = jnp.sum(
            weights[:, None, None] * noise_clipped, axis=0
        )  # (H, nu)
        new_mean = running.u_mean + weighted_noise * self.config.gamma_mean

        # --- 8. Adaptive covariance (diagonal, per-action) ---
        delta = perturbed - new_mean[None, :, :]  # (K, H, nu)
        # Weighted variance per action dim, averaged across horizon
        weighted_var = jnp.sum(weights[:, None, None] * delta ** 2, axis=0)  # (H, nu)
        cov_update = jnp.mean(weighted_var, axis=0)  # (nu,) mean across horizon

        new_cov = jnp.where(
            self.config.gamma_sigma > 0,
            (1 - self.config.gamma_sigma) * running.cov_diag + self.config.gamma_sigma * cov_update + self.config.cov_floor,
            running.cov_diag,
        )

        # --- 9. Warm-start: shift mean forward, repeat last ---
        shifted_mean = jnp.concatenate([new_mean[1:], new_mean[-1:]], axis=0)

        # --- 10. Extract action and optimal trajectory ---
        u = new_mean[0]

        # Optimal trajectory: rollout with mean actions
        _, optim_states = jax.lax.scan(
            scan_step, obs[None, :].squeeze(0)[None, :].reshape(1, -1).repeat(1, axis=0),
            new_mean[:, None, :].squeeze(1)[None, :, :].swapaxes(0, 1),
        )
        # Simpler: just do a single rollout with mean
        single_state = obs
        def single_scan(state, action):
            state_batch = state[None, :]
            action_batch = action[None, :]
            next_state, _ = self.rollout_fn(None, state_batch, action_batch, dyn_params, False)
            return next_state[0], next_state[0]
        _, optim_traj_states = jax.lax.scan(single_scan, single_state, new_mean)
        optim_traj = jnp.concatenate([obs[None, :], optim_traj_states], axis=0)  # (H+1, num_obs)

        new_running = MPPITorchRunning(
            u_mean=shifted_mean,
            cov_diag=new_cov,
            beta=new_beta,
            key=key,
        )

        # Extract diverse sampled trajectories for visualization
        n_viz = 10
        sorted_indices = jnp.argsort(costs)
        step_size = max(K // n_viz, 1)  # static int for JIT
        viz_indices = sorted_indices[::step_size][:n_viz]
        sampled_trajs = state_traj[:, viz_indices, :]  # (H+1, n_viz, num_obs)

        info = {
            'trajectory': optim_traj,
            'sampled_trajectories': sampled_trajs,
            'a_mean_jnp': new_mean,
            'action': None,
            'action_candidate': None,
            'x_all': None,
            'y_all': None,
        }

        return u, new_running, info
