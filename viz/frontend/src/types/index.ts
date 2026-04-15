export interface Obstacle {
  id: string;
  type: 'rect' | 'circle';
  // rect: world coords (min/max)
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  // circle: world coords
  cx?: number;
  cy?: number;
  r?: number;
}

export interface Pose {
  x: number;
  y: number;
  heading: number;
}

export interface StepData {
  step: number;
  state: number[];
  action: number[];
  pos_error: number;
  heading_error: number;
  predicted_trajectory: number[][] | null;
  sampled_trajectories: number[][][] | null;
  goal_reward_triggered: boolean;
}

export interface SdfData {
  grid: number[][];
  origin: number[];
  resolution: number;
  full_shape: number[];
}

export interface DynamicsInfo {
  state_dims: number;
  action_dims: number;
  kwargs: Record<string, number>;
}

export interface Preset {
  scene: {
    boxes: number[][];
    circles: number[][];
    start: number[];
    goal: number[];
  };
  dynamics: string;
  kwargs: Record<string, number>;
  dt: number;
  n_rollouts: number;
  sigma: number;
  steer_sigma: number;
  lam: number;
  horizon_knots: number;
  num_intermediate: number;
  goal_pos_weight: number;
  goal_heading_weight: number;
  goal_vel_weight?: number;
  collision_weight: number;
  collision_decay: number;
  collision_inside_weight: number;
  smoothness_weight: number;
  terminal_weight?: number;
  control_reg_weight?: number;
  sign_change_weight?: number;
  goal_tolerance: number;
  max_steps: number;
  backend?: string;
  gamma_sigma?: number;
  cov_floor?: number;
  stop_threshold?: number;
  use_global_planner?: boolean;
  global_planner_type?: string;
  replan_interval?: number;
}

export interface WorldBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}
