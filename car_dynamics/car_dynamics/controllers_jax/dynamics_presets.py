"""
Ready-made dynamics models for common robot bases.

Each factory returns a rollout_fn matching the MPPI interface:
    rollout_fn(obs_history, state, action, dynamic_params_tuple, debug) -> (next_state, {})

These are simple kinematic models suitable for local planning. For learned or
high-fidelity dynamics, use the existing DBM or Transformer rollout paths.
"""

import jax
import jax.numpy as jnp


def holonomic_rollout_fn(dt):
    """
    Holonomic (omnidirectional) base — e.g., RoboCasa PandaOmron.

    State:  [x, y, θ]       (3D)
    Action: [v_fwd, v_side, ω]  (body-frame velocities)

    Dynamics (Euler integration):
        x' = x + (v_fwd * cos(θ) - v_side * sin(θ)) * dt
        y' = y + (v_fwd * sin(θ) + v_side * cos(θ)) * dt
        θ' = θ + ω * dt
    """
    @jax.jit
    def rollout_fn(obs_history, state, action, dynamic_params_tuple, debug=False):
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        vf, vs, omega = action[:, 0], action[:, 1], action[:, 2]
        c, s = jnp.cos(theta), jnp.sin(theta)
        next_x = x + (vf * c - vs * s) * dt
        next_y = y + (vf * s + vs * c) * dt
        next_theta = theta + omega * dt
        return jnp.stack([next_x, next_y, next_theta], axis=1), {}

    return rollout_fn


def unicycle_rollout_fn(dt, wheelbase=0.3):
    """
    Unicycle / Ackermann steering — e.g., simple car, TurtleBot with steering.

    State:  [x, y, θ, v]    (4D)
    Action: [accel, steer]   (2D)

    Dynamics (Euler integration):
        x' = x + v * cos(θ) * dt
        y' = y + v * sin(θ) * dt
        θ' = θ + v * tan(steer) / wheelbase * dt
        v' = v + accel * dt
    """
    @jax.jit
    def rollout_fn(obs_history, state, action, dynamic_params_tuple, debug=False):
        x, y, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        accel, steer = action[:, 0], action[:, 1]
        next_x = x + v * jnp.cos(theta) * dt
        next_y = y + v * jnp.sin(theta) * dt
        next_theta = theta + v * jnp.tan(steer) / wheelbase * dt
        next_v = v + accel * dt
        return jnp.stack([next_x, next_y, next_theta, next_v], axis=1), {}

    return rollout_fn


def diff_drive_rollout_fn(dt, wheel_separation=0.3):
    """
    Differential drive — e.g., TurtleBot, Hello Robot Stretch.

    State:  [x, y, θ]           (3D)
    Action: [v_left, v_right]    (2D, wheel velocities)

    Dynamics (Euler integration):
        v     = (v_right + v_left) / 2
        omega = (v_right - v_left) / wheel_separation
        x' = x + v * cos(θ) * dt
        y' = y + v * sin(θ) * dt
        θ' = θ + omega * dt
    """
    @jax.jit
    def rollout_fn(obs_history, state, action, dynamic_params_tuple, debug=False):
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        vl, vr = action[:, 0], action[:, 1]
        v = (vr + vl) / 2.0
        omega = (vr - vl) / wheel_separation
        next_x = x + v * jnp.cos(theta) * dt
        next_y = y + v * jnp.sin(theta) * dt
        next_theta = theta + omega * dt
        return jnp.stack([next_x, next_y, next_theta], axis=1), {}

    return rollout_fn


def robocasa_holonomic_rollout_fn(
    coef_dx_mm=None, coef_dy_mm=None, coef_dyaw_rad=None,
    # Legacy fallback
    dt_fwd=0.037, dt_side=0.019, dt_yaw=0.066,
    **_ignored,
):
    """
    Calibrated holonomic base for RoboCasa PandaOmron.

    Uses a polynomial model with cubic terms + cross-coupling,
    fitted on 125 combined-action measurements (5x5x5 grid).

    Features: [vf, vf^3, vs, vs^3, om, om^3, vf*vs, vf*om, vs*om, vf*vs*om]
    Outputs: dx_body (mm), dy_body (mm), dyaw (rad)

    Validated on 50 random [-1,1] actions:
      Position RMSE: 1.68 mm, Max: 3.11 mm
      Yaw RMSE: 0.11 deg

    State:  [x, y, θ]       (3D)
    Action: [v_fwd, v_side, ω]  (body-frame, [-1, 1])
    """
    if coef_dx_mm is not None:
        _cx = jnp.array(coef_dx_mm, dtype=jnp.float32)
        _cy = jnp.array(coef_dy_mm, dtype=jnp.float32)
        _cyaw = jnp.array(coef_dyaw_rad, dtype=jnp.float32)
        _use_poly = True
    else:
        _use_poly = False

    @jax.jit
    def rollout_fn(obs_history, state, action, dynamic_params_tuple, debug=False):
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        vf, vs, om = action[:, 0], action[:, 1], action[:, 2]

        if _use_poly:
            # Polynomial features: [vf, vf^3, vs, vs^3, om, om^3, vf*vs, vf*om, vs*om, vf*vs*om]
            feats = jnp.stack([
                vf, vf**3,
                vs, vs**3,
                om, om**3,
                vf * vs, vf * om, vs * om,
                vf * vs * om,
            ], axis=1)  # (N, 10)

            dx_body = feats @ _cx / 1000.0  # mm → m
            dy_body = feats @ _cy / 1000.0
            dtheta = feats @ _cyaw
        else:
            dx_body = vf * dt_fwd
            dy_body = vs * dt_side
            dtheta = om * dt_yaw

        c, s = jnp.cos(theta), jnp.sin(theta)
        next_x = x + (dx_body * c - dy_body * s)
        next_y = y + (dx_body * s + dy_body * c)
        next_theta = theta + dtheta
        return jnp.stack([next_x, next_y, next_theta], axis=1), {}

    return rollout_fn


# Registry: name -> (factory_fn, num_obs, num_actions)
# factory_fn signature: factory_fn(dt, **kwargs) -> rollout_fn
DYNAMICS_REGISTRY = {
    'holonomic': {
        'factory': lambda dt, **kw: holonomic_rollout_fn(dt),
        'num_obs': 3,
        'num_actions': 3,
    },
    'robocasa_holonomic': {
        'factory': lambda dt, **kw: robocasa_holonomic_rollout_fn(**{
            k: kw[k] for k in kw
            if k in ('coef_dx_mm','coef_dy_mm','coef_dyaw_rad',
                     'dt_fwd','dt_side','dt_yaw')
        }),
        'num_obs': 3,
        'num_actions': 3,
    },
    'unicycle': {
        'factory': lambda dt, **kw: unicycle_rollout_fn(dt, kw.get('wheelbase', 0.3)),
        'num_obs': 4,
        'num_actions': 2,
    },
    'diff_drive': {
        'factory': lambda dt, **kw: diff_drive_rollout_fn(dt, kw.get('wheel_separation', 0.3)),
        'num_obs': 3,
        'num_actions': 2,
    },
}
