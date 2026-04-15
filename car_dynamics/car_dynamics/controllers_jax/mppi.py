from dataclasses import dataclass
import jax
import numpy as np
import jax.numpy as jnp
import time
from termcolor import colored
from .base import BaseController
from functools import partial
import flax
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

@dataclass
class MPPIParams:
    sigma: float
    gamma_mean: float
    gamma_sigma: float
    discount: float
    sample_sigma: float
    lam: float
    n_rollouts: int
    h_knot: int
    a_min: jnp.ndarray
    a_max: jnp.ndarray
    a_mag: jnp.ndarray
    a_shift: jnp.ndarray
    delay: int
    len_history: int
    debug: bool
    fix_history: bool
    num_obs: int
    num_actions: int
    num_intermediate: int
    spline_order: int
    smooth_alpha: float = 0.8
    dynamics: str = 'dbm'
    dual: bool = False
    # Adaptive covariance (mppi_torch-style)
    cov_floor: float = 0.0           # kappa: minimum covariance diagonal. 0 = off.
    # Adaptive temperature (mppi_torch-style)
    adaptive_beta: bool = False      # enable adaptive temperature tuning
    beta_eta_upper: float = 100.0    # eta > this → sharpen (fewer effective samples needed)
    beta_eta_lower: float = 10.0     # eta < this → soften (too few effective samples)
    beta_shrink: float = 0.9         # multiplier when sharpening
    beta_grow: float = 1.2           # multiplier when softening

@flax.struct.dataclass
class MPPIRunningParams:
    a_mean: jnp.ndarray
    a_cov: jnp.ndarray
    prev_a: jnp.ndarray
    state_hist: jnp.ndarray
    key: jax.random.PRNGKey
    beta: float = 0.05  # adaptive temperature (initialized from params.lam)

def slow_scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)
    
    
class MPPIController(BaseController):
    def __init__(self,
                params: MPPIParams, rollout_fn: callable, rollout_start_fn: callable, key):
        """ MPPI implemented in Jax """
        self.params = params
        self.rollout_fn = rollout_fn
        self.rollout_start_fn = rollout_start_fn
        self._init_buffers()
        
        if params.dynamics in ('dbm', 'scan'):
            self.scan_fn = jax.lax.scan
            self._get_rollout = self._get_rollout_dbm
        elif params.dynamics == 'transformer-jax':
            self.scan_fn = self.scan_seq_jax
            self._get_rollout = self._get_rollout_nn
        else:
            raise ValueError(f"Unknown dynamics model: {params.dynamics}")
        
        
    def _init_buffers(self, ):
        self.spline_order = self.params.spline_order
        self.H = (self.params.h_knot -1 ) * self.params.num_intermediate + 1 
        self.a_mean = jnp.zeros((self.H, self.params.num_actions))
        s = self.params.sigma
        sigmas = jnp.array(s) if hasattr(s, '__len__') else jnp.array([s] * self.params.num_actions)
        a_cov_per_step = jnp.diag(sigmas ** 2)
        self.a_cov = jnp.tile(a_cov_per_step[None, :, :], (self.H, 1, 1))
        self.a_mean_init = self.a_mean[-1:]
        self.a_cov_init = self.a_cov[-1:]
        
        self.step_us = jnp.arange(self.H)
        self.step_nodes = jnp.arange(self.params.h_knot) * (self.params.num_intermediate)
        self.prev_a = jnp.zeros((self.params.delay, self.params.num_actions))
        # self.step_count = 0
        
        state_hist_len = max(self.params.len_history, 1)
        self.state_hist_init = jnp.zeros((state_hist_len, self.params.num_obs + self.params.num_actions))
        
        self.action_sampled = jnp.zeros((self.params.n_rollouts, self.H, self.params.num_actions))
        self.action_init_buf = jnp.zeros((self.params.n_rollouts, self.H + self.params.delay, self.params.num_actions))
        self.state_init_buf = jnp.ones((self.params.num_obs,))
        self.x_all = []
        self.y_all = []
        
        self.node2u_vmap = jax.vmap(self.node2u, in_axes=(0,))
        self.u2node_vmap = jax.vmap(self.u2node, in_axes=(0,))
        
    @partial(jax.jit, static_argnums=(0,))
    def node2u(self, nodes):
        spline = InterpolatedUnivariateSpline(self.step_nodes, nodes, k=self.spline_order)
        us = spline(self.step_us)
        return us
    
    @partial(jax.jit, static_argnums=(0,))
    def u2node(self, us):
        spline = InterpolatedUnivariateSpline(self.step_us, us, k=self.spline_order)
        nodes = spline(self.step_nodes)
        return nodes
    
    def get_init_params(self, ):
        return MPPIRunningParams(
            a_mean = self.a_mean,
            a_cov = self.a_cov,
            prev_a = self.prev_a,
            key = jax.random.PRNGKey(123),
            state_hist = self.state_hist_init,
            beta = self.params.lam,
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def _running_average(self, carry, x):
        prev_x = carry
        new_x = x * self.params.smooth_alpha + prev_x * (1 - self.params.smooth_alpha)
        return new_x, new_x
    
    @partial(jax.jit, static_argnums=(0,))
    def normalize_action(self, a_sampled):
        for d in range(len(self.params.a_min)):
            a_sampled = a_sampled.at[:, :, d].set(jnp.clip(a_sampled[:, :, d], self.params.a_min[d], self.params.a_max[d]) * self.params.a_mag[d] + self.params.a_shift[d])
        return a_sampled
    
    @partial(jax.jit, static_argnums=(0,))
    def _rollout_jit(self, carry, action):
        state, obs_history, dynamic_params_tuple = carry
        obs_history = obs_history.at[-1, :self.params.n_rollouts, :self.params.num_obs].set(state[:, :self.params.num_obs])
        obs_history = obs_history.at[-1, :self.params.n_rollouts, -self.params.num_actions:].set(action)
        state, _ = self.rollout_fn(obs_history, state, action, dynamic_params_tuple, self.params.debug)
        obs_history = obs_history.at[:-1].set(obs_history[1:])    
        return (state, obs_history, dynamic_params_tuple), state
    
    def scan_seq(self, rollout_fn, init, action_list):
        state, obs_history, dynamic_params_tuple  = init
        obs_history = obs_history.at[-1, :self.params.n_rollouts, :self.params.num_obs].set(state[:, :self.params.num_obs])
        obs_history = obs_history.at[-1, :self.params.n_rollouts, -self.params.num_actions:].set(action_list[0])
        obs_history = jnp.swapaxes(obs_history, 0, 1)
        action_list = jnp.swapaxes(action_list, 0, 1)
        state_list, debug_info = self.rollout_fn(obs_history, state, action_list, dynamic_params_tuple, self.params.debug)
        return debug_info, state_list
    
    # @partial(jax.jit, static_argnums=(0,))
    def scan_seq_jax(self, key, rollout_fn, init, action_list):
        state, obs_history, dynamic_params_tuple  = init
        obs_history = obs_history.at[-1, :self.params.n_rollouts, :self.params.num_obs].set(state[:, :self.params.num_obs])
        obs_history = obs_history.at[-1, :self.params.n_rollouts, -self.params.num_actions:].set(action_list[0])
        obs_history = jnp.swapaxes(obs_history, 0, 1)
        action_list = jnp.swapaxes(action_list, 0, 1)
        key, key2 = jax.random.split(key, 2)
        state_list, debug_info = self.rollout_fn(key2, obs_history, state, action_list, dynamic_params_tuple, self.params.debug)
        return debug_info, state_list
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_rollout_dbm(self, key, state_init, state_hist, actions, dynamic_params_tuple, fix_history=False):
        """
        Get rollout for dynamic bicycle model.

        This method prepares the input data, calls the rollout function, and processes
        the output for the dynamics-based model. It handles the initialization of states,
        observation history, and actions, then performs the rollout using the scan function.

        Args:
            key (jax.random.PRNGKey): Random key for JAX operations.
            state_init (jnp.ndarray): Initial state.
            state_hist (jnp.ndarray): History of previous states.
            actions (jnp.ndarray): Sampled actions for rollouts.
            dynamic_params_tuple (tuple): Parameters for the dynamics model.
            fix_history (bool, optional): Whether to fix the history. Defaults to False.

        Returns:
            jnp.ndarray: Array of states for all timesteps in the rollout.
        """
        n_rollouts = actions.shape[0]
        state = jnp.tile(jnp.expand_dims(state_init, 0), (n_rollouts, 1))
        state_list = state[None]
        obs_history = jnp.tile(jnp.expand_dims(state_hist.copy(), 0), (n_rollouts, 1, 1))
        self.rollout_start_fn()
        obs_history = jnp.swapaxes(obs_history, 0, 1)
        actions = jnp.swapaxes(actions, 0, 1)
        # For 1-step model
        _, state_list2 = self.scan_fn(self._rollout_jit, (state, obs_history, dynamic_params_tuple), actions)
        state_list = jnp.concatenate((state_list, state_list2), axis=0)
        state_list_jnp = jnp.array(state_list)
        return state_list_jnp
    
    
    def _get_rollout_nn(self, key, state_init, state_hist, actions, dynamic_params_tuple, fix_history=False):
        """
        Get rollout for neural network-based dynamics model.

        This method prepares the input data, calls the rollout function, and processes
        the output for the neural network-based dynamics model.

        Args:
            key (jax.random.PRNGKey): Random key for JAX operations.
            state_init (jnp.ndarray): Initial state.
            state_hist (jnp.ndarray): History of previous states.
            actions (jnp.ndarray): Sampled actions for rollouts.
            dynamic_params_tuple (tuple): Parameters for the dynamics model.
            fix_history (bool, optional): Whether to fix the history. Defaults to False.

        Returns:
            jnp.ndarray: Array of states for all timesteps in the rollout.
        """
        n_rollouts = actions.shape[0]
        state = jnp.tile(jnp.expand_dims(state_init, 0), (n_rollouts, 1))
        state_list = state[None]
        obs_history = jnp.tile(jnp.expand_dims(state_hist.copy(), 0), (n_rollouts, 1, 1))
        self.rollout_start_fn()
        obs_history = jnp.swapaxes(obs_history, 0, 1)
        actions = jnp.swapaxes(actions, 0, 1)
        _, state_list2 = self.scan_fn(key, self._rollout_jit, (state, obs_history, dynamic_params_tuple), actions)
        state_list = jnp.concatenate((state_list, state_list2), axis=0)
        state_list_jnp = jnp.array(state_list)
        return state_list_jnp
    

    
    @partial(jax.jit, static_argnums=(0,))
    def single_step_reward(self, carry, pair):
        """
        Calculate the reward for a single step in the rollout.

        Args:
            carry (tuple): Contains the current step and previous action.
            pair (tuple): Contains the current state, action, and goal for this step.

        Returns:
            tuple: Updated carry (next step and current action) and the calculated reward.
        """
        step, prev_action = carry
        state_step, action_step, goal = pair

        dist_pos = jnp.linalg.norm(state_step[:, :2] - goal[:2], axis=1)
        diff_psi = state_step[:, 2] - goal[2]
        diff_psi = jnp.arctan2(jnp.sin(diff_psi), jnp.cos(diff_psi))
        diff_vel = state_step[:, 3] - goal[3]
        
        reward_pos_err = -dist_pos ** 2
        reward_psi = -diff_psi ** 2
        reward_vel = -diff_vel ** 2
        reward_throttle = - (action_step[:, 0] - prev_action[:, 0]) ** 2
        reward = reward_pos_err * 5.0 + reward_psi * 5.0 + reward_vel * 1. + reward_throttle * 0.0
        reward *= (self.params.discount ** step)
        return (step + 1, action_step), reward
    
    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, state, action, goal_list):
        """
        Calculate the total reward for each rollout trajectory.

        Args:
            state (jnp.ndarray): State trajectory for all rollouts.
            action (jnp.ndarray): Action trajectory for all rollouts.
            goal_list (jnp.ndarray): List of goal states for each timestep.

        Returns:
            jnp.ndarray: Total rewards for each rollout.
        """
        actions = jnp.swapaxes(action, 0, 1)
        _, reward_list = jax.lax.scan(self.single_step_reward, (0, actions[0]), (state[1:], actions, goal_list[1:]))
        rewards = jnp.sum(reward_list, axis=0)
        return rewards
    
    
    @partial(jax.jit, static_argnums=(0,)) 
    def feed_hist(self, param: MPPIRunningParams, obs, action):
        """
        Update the state history with the latest observation and action.

        Args:
            param (MPPIRunningParams): Current MPPI running parameters.
            obs (jnp.ndarray): Latest observation.
            action (jnp.ndarray): Latest action.

        Returns:
            MPPIRunningParams: Updated MPPI running parameters with new state history.
        """
        state = jnp.array(obs[:self.params.num_obs])
        action_tensor = jnp.array(action[:self.params.num_actions])
        state_hist = param.state_hist
        state_hist = state_hist.at[-1, :self.params.num_obs].set(state)
        state_hist = state_hist.at[-1, self.params.num_obs:self.params.num_obs + self.params.num_actions].set(action_tensor)
        state_hist = state_hist.at[:-1].set(state_hist[1:])
        
        return MPPIRunningParams(
            a_mean = param.a_mean,
            a_cov = param.a_cov,
            prev_a = param.prev_a,
            state_hist = state_hist,
            key = param.key,
            beta = param.beta,
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def debug_rollout(self, key, obs, a_list, running_params: MPPIRunningParams, dynamic_params_tuple):
        """
        Perform a debug rollout to visualize the optimized trajectory.

        This function executes a single rollout using the current mean action sequence
        and returns the resulting state trajectory. It's useful for debugging and
        visualizing the behavior of the controller.

        Args:
            key (jax.random.PRNGKey): Random key for any stochastic operations.
            obs (jnp.ndarray): Current observation of the system state.
            a_list (jnp.ndarray): Mean action sequence to use for the rollout.
            running_params (MPPIRunningParams): Current MPPI running parameters.
            dynamic_params_tuple (Tuple): Parameters for the dynamics model.

        Returns:
            jnp.ndarray: Optimized state trajectory resulting from the rollout.
        """
        state_init = jnp.array(obs)
        action_expand = jnp.tile(jnp.expand_dims(a_list, 0), (self.params.n_rollouts, 1, 1))
        key, key2 = jax.random.split(key, 2)
        optim_traj = jnp.stack(self._get_rollout(key2, state_init, running_params.state_hist, action_expand, dynamic_params_tuple, self.params.fix_history))[:, 0]
        return optim_traj
        
        
        
    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        obs,
        goal_list, 
        running_params: MPPIRunningParams,
        dynamic_params_tuple,
    ):
        """
        Execute the Model Predictive Path Integral (MPPI) control algorithm.

        This function performs the core MPPI computation, including:
        1. Sampling action trajectories
        2. Simulating rollouts using the dynamics model
        3. Evaluating costs for each trajectory
        4. Updating the control distribution based on trajectory costs

        Args:
            obs (jnp.ndarray): Current observation of the system state.
            goal_list (List[jnp.ndarray]): List of goal states for the trajectory.
            running_params (MPPIRunningParams): Current MPPI running parameters.
            dynamic_params_tuple (Tuple): Parameters for the dynamics model.
            vis_optim_traj (bool, optional): Flag to visualize the optimized trajectory. Defaults to False.
            vis_all_traj (bool, optional): Flag to visualize all sampled trajectories. Defaults to False.

        Returns:
            Tuple: Contains the following elements:
                - jnp.ndarray: Optimal action to take at the current timestep.
                - MPPIRunningParams: Updated MPPI running parameters.
                - Dict: Additional information for debugging and visualization.
        """
        
        ## Note: 1. Sampling action trajectories
        key_use, self_key = jax.random.split(running_params.key, 2)
        key_use = jax.random.split(key_use, self.params.n_rollouts)
        
        def single_sample(key, traj_mean, traj_cov):
            keys = jax.random.split(key, self.params.h_knot)
            return jax.vmap(
                lambda key, mean, cov: jax.random.multivariate_normal(key, mean, cov)
            )(keys, traj_mean, traj_cov)

        a_mean_waypoint = running_params.a_mean[::self.params.num_intermediate]
        
        ## Spline interpolation
        for d in range(self.params.num_actions):
            a_mean_waypoint = a_mean_waypoint.at[:, d].set(self.u2node(running_params.a_mean[:, d]))
        
        
        a_cov_waypoint = running_params.a_cov[::self.params.num_intermediate]
        
        a_sampled_waypoint = jax.vmap(single_sample, in_axes=(0, None, None))( # (N, h_knot, action_dim)
            key_use, a_mean_waypoint, a_cov_waypoint,
        )
    
        ### Spline interpolation
        a_sampled = self.action_sampled.copy()
        for d in range(self.params.num_actions):
            a_sampled = a_sampled.at[:, :, d].set(self.node2u_vmap(a_sampled_waypoint[:, :, d]))
        

        a_sampled_raw = self.normalize_action(a_sampled)
        a_sampled = self.action_init_buf.copy()
        a_sampled = a_sampled.at[:, :self.params.delay, :].set(running_params.prev_a)
        a_sampled = a_sampled.at[:, self.params.delay:, :].set(a_sampled_raw)
        
        state_init = self.state_init_buf.copy()
        for i_ in range(self.params.num_obs):
            state_init = state_init.at[i_].set(state_init[i_] * obs[i_])
        
        ## Note: 2. Simulating rollouts using the dynamics model
        self_key, key2 = jax.random.split(self_key, 2)
        state_list = self._get_rollout(key2, state_init, running_params.state_hist, a_sampled, dynamic_params_tuple, self.params.fix_history)   # List
        

        reward_rollout = self.get_reward(state_list, a_sampled, goal_list)

        # Sparse goal-reach bonus: position + heading + stopped
        # Exponential decay: earlier reach = higher reward
        if hasattr(self, '_grr') and self._grr > 0:
            goal_pos = goal_list[-1, :2]
            goal_heading = goal_list[-1, 2]
            # Position: within threshold
            dist_to_goal = jnp.linalg.norm(
                state_list[:, :, :2] - goal_pos[None, None, :], axis=2
            )  # (H+1, n_rollouts)
            pos_ok = dist_to_goal < self._grt
            # Heading: within threshold * 5
            heading_diff = jnp.abs(jnp.arctan2(
                jnp.sin(state_list[:, :, 2] - goal_heading),
                jnp.cos(state_list[:, :, 2] - goal_heading),
            ))
            heading_ok = heading_diff < (self._grt * 5.0)
            # Velocity: must be nearly stopped (no drive-through)
            if self.params.num_obs >= 4:
                vel_ok = jnp.abs(state_list[:, :, 3]) < 0.3
            else:
                vel_ok = jnp.ones_like(pos_ok)  # no velocity state, skip check
            reached_mask = pos_ok & heading_ok & vel_ok
            first_step = jnp.argmax(reached_mask, axis=0)
            ever_reached = jnp.any(reached_mask, axis=0).astype(jnp.float32)
            H_float = float(state_list.shape[0] - 1)
            reward_bonus = ever_reached * self._grr * jnp.exp(-3.0 * first_step / H_float)
            reward_rollout = reward_rollout + reward_bonus

        cost_rollout = -reward_rollout

        ## Note: 3. Cost weighting with adaptive temperature
        beta = running_params.beta
        cost_exp = jnp.exp(-(cost_rollout - jnp.min(cost_rollout)) / beta)
        eta = jnp.sum(cost_exp)  # effective sample count (partition function)
        weight = cost_exp / eta

        # Adaptive beta (mppi_torch-style): tune temperature based on eta
        # Clamp to [lam/10, lam*20] to prevent explosion
        beta_min = self.params.lam * 0.1
        beta_max = self.params.lam * 20.0
        new_beta = jnp.where(
            self.params.adaptive_beta,
            jnp.clip(
                jnp.where(eta > self.params.beta_eta_upper, beta * self.params.beta_shrink,
                jnp.where(eta < self.params.beta_eta_lower, beta * self.params.beta_grow,
                beta)),
                beta_min, beta_max,
            ),
            beta,
        )

        a_sampled = a_sampled[:, self.params.delay:, :]

        ## Note: 4. Mean update (weighted average with momentum)
        a_mean = jnp.sum(
            weight[:, None, None] * a_sampled, axis=0
        ) * self.params.gamma_mean + running_params.a_mean * (
            1 - self.params.gamma_mean
        )

        ## Note: 5. Covariance update (adaptive diagonal with floor)
        a_cov = jnp.sum(
                        weight[:, None, None, None] * ((a_sampled - a_mean)[..., None] * (a_sampled - a_mean)[:, :, None, :]),
                        axis=0,
                    ) * self.params.gamma_sigma + running_params.a_cov * (1 - self.params.gamma_sigma)

        # Add covariance floor (kappa) to prevent sigma collapse
        a_cov = a_cov + self.params.cov_floor * jnp.eye(self.params.num_actions)[None, :, :]

        u = a_mean[0]

        optim_traj = None
        action_expand = jnp.tile(jnp.expand_dims(a_mean, 0), (self.params.n_rollouts, 1, 1))

        self_key, key2 = jax.random.split(self_key, 2)
        optim_traj = jnp.stack(self._get_rollout(key2, state_init, running_params.state_hist, action_expand, dynamic_params_tuple, self.params.fix_history))[:, 0]

        prev_a = jnp.concatenate([running_params.prev_a[1:], a_mean[:1]], axis=0)

        new_running_params = MPPIRunningParams(
            a_mean = jnp.concatenate([a_mean[1:], a_mean[-1:]], axis=0),
            a_cov = jnp.concatenate([a_cov[1:], a_cov[-1:]], axis=0),
            state_hist = running_params.state_hist,
            prev_a = prev_a,
            key = self_key,
            beta = new_beta,
        )

        # Extract diverse sampled trajectories for visualization
        # Evenly spaced across cost-sorted rollouts (best → worst)
        n_viz = 10
        sorted_indices = jnp.argsort(cost_rollout)
        step_size = max(self.params.n_rollouts // n_viz, 1)  # static int for JIT
        viz_indices = sorted_indices[::step_size][:n_viz]
        # state_list: (H+1, n_rollouts, num_obs)
        sampled_trajs = state_list[:, viz_indices, :]  # (H+1, n_viz, num_obs)

        info_dict = {
            'trajectory': optim_traj,
            'sampled_trajectories': sampled_trajs,
            'action': None,
            'a_mean_jnp': a_mean,
            'action_candidate': None, 'x_all': None, 'y_all': None,
        } 
        
        return u,  new_running_params,  info_dict
