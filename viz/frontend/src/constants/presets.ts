import type { Preset } from '../types';

export const PRESETS: Record<string, Preset> = {
  straight: {
    scene: {
      boxes: [], circles: [],
      start: [-2.0, 0.0, 0.0],
      goal: [2.0, 0.0, 0.0],
    },
    dynamics: 'unicycle', kwargs: { wheelbase: 0.3 },
    dt: 0.15, n_rollouts: 4096, sigma: 0.15, steer_sigma: 0.001, lam: 0.05,
    horizon_knots: 4, num_intermediate: 18,
    goal_pos_weight: 5000.0, goal_heading_weight: 200.0, goal_vel_weight: 100.0,
    collision_weight: 0.0, collision_decay: 3.0,
    collision_inside_weight: 0.0, smoothness_weight: 10.0,
    goal_tolerance: 0.05, max_steps: 400,
  },
  lane_change: {
    scene: {
      boxes: [[-0.3, -0.6, 0.3, 0.6]],
      circles: [],
      start: [-2.0, 0.0, 0.0],
      goal: [2.0, 0.0, 0.0],
    },
    dynamics: 'unicycle', kwargs: { wheelbase: 0.3 },
    dt: 0.15, n_rollouts: 4096, sigma: 0.15, steer_sigma: 0.03, lam: 0.05,
    horizon_knots: 12, num_intermediate: 6,
    goal_pos_weight: 5000.0, goal_heading_weight: 200.0, goal_vel_weight: 100.0,
    collision_weight: 30.0, collision_decay: 3.0,
    collision_inside_weight: 300.0, smoothness_weight: 30.0,
    goal_tolerance: 0.05, max_steps: 400,
  },
  s_curve: {
    scene: {
      boxes: [[-0.8, -0.3, 0.0, 2.0], [0.0, -2.0, 0.8, 0.3]],
      circles: [],
      start: [-2.0, 1.0, 0.0],
      goal: [2.0, -1.0, 0.0],
    },
    dynamics: 'unicycle', kwargs: { wheelbase: 0.3 },
    dt: 0.15, n_rollouts: 4096, sigma: 0.15, steer_sigma: 0.03, lam: 0.05,
    horizon_knots: 12, num_intermediate: 6,
    goal_pos_weight: 5000.0, goal_heading_weight: 200.0, goal_vel_weight: 100.0,
    collision_weight: 30.0, collision_decay: 3.0,
    collision_inside_weight: 300.0, smoothness_weight: 30.0,
    goal_tolerance: 0.05, max_steps: 500,
  },
  slalom: {
    scene: {
      boxes: [],
      circles: [[-1.0, 0.8, 0.25], [0.0, -0.8, 0.25], [1.0, 0.8, 0.25]],
      start: [-2.0, 0.0, 0.0],
      goal: [2.0, 0.0, 0.0],
    },
    dynamics: 'unicycle', kwargs: { wheelbase: 0.3 },
    dt: 0.15, n_rollouts: 4096, sigma: 0.15, steer_sigma: 0.03, lam: 0.05,
    horizon_knots: 12, num_intermediate: 6,
    goal_pos_weight: 5000.0, goal_heading_weight: 200.0, goal_vel_weight: 100.0,
    collision_weight: 30.0, collision_decay: 3.0,
    collision_inside_weight: 300.0, smoothness_weight: 30.0,
    goal_tolerance: 0.05, max_steps: 500,
  },
};
