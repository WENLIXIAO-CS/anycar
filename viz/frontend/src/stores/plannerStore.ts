import { create } from 'zustand';
import type { DynamicsInfo } from '../types';
import { PRESETS } from '../constants/presets';

export interface PlannerState {
  // Dynamics
  dynamics: string;
  dynamicsKwargs: Record<string, number>;
  dynamicsInfo: Record<string, DynamicsInfo>;

  // MPPI params
  dt: number;
  nRollouts: number;
  sigma: number;
  lam: number;
  horizonKnots: number;
  numIntermediate: number;
  steerSigma: number;
  backend: string;
  gammaSigma: number;
  covFloor: number;

  // Cost weights
  goalPosWeight: number;
  goalHeadingWeight: number;
  goalVelWeight: number;
  collisionWeight: number;
  collisionDecay: number;
  collisionInsideWeight: number;
  smoothnessWeight: number;
  terminalWeight: number;
  controlRegWeight: number;
  signChangeWeight: number;
  goalReachReward: number;
  goalReachThreshold: number;

  // Map settings
  resolution: number;
  inflateRadius: number;
  useGlobalPlanner: boolean;
  globalPlannerType: string;

  // Convergence
  goalTolerance: number;
  stopThreshold: number;
  maxSteps: number;
  replanInterval: number;

  // UI state
  planning: boolean;
  playing: boolean;
  canPlay: boolean;
  statusText: string;
  currentStep: number;
  posError: number | null;
  yawError: number | null;
  goalRewardTriggered: boolean;

  // Actions
  setParam: <K extends keyof PlannerState>(key: K, value: PlannerState[K]) => void;
  fetchDynamicsInfo: () => Promise<void>;
  updateDynamicsKwargs: () => void;
  loadPreset: (name: string) => void;
  updateStepInfo: (step: number, posErr: number, yawErr: number, goalReward: boolean) => void;
  buildQueryParams: () => URLSearchParams;
}

export const usePlannerStore = create<PlannerState>((set, get) => ({
  // Dynamics
  dynamics: 'holonomic',
  dynamicsKwargs: {},
  dynamicsInfo: {},

  // MPPI params
  dt: 0.05,
  nRollouts: 2048,
  sigma: 0.15,
  lam: 0.005,
  horizonKnots: 10,
  numIntermediate: 8,
  steerSigma: 0.03,
  backend: 'spline',
  gammaSigma: 0.5,
  covFloor: 0.001,

  // Cost weights
  goalPosWeight: 5000.0,
  goalHeadingWeight: 200.0,
  goalVelWeight: 100.0,
  collisionWeight: 30.0,
  collisionDecay: 3.0,
  collisionInsideWeight: 300.0,
  smoothnessWeight: 30.0,
  terminalWeight: 10.0,
  controlRegWeight: 0.1,
  signChangeWeight: 50.0,
  goalReachReward: 0.0,
  goalReachThreshold: 0.1,

  // Map settings
  resolution: 0.05,
  inflateRadius: 0.15,
  useGlobalPlanner: true,
  globalPlannerType: 'rrt',

  // Convergence
  goalTolerance: 0.05,
  stopThreshold: 0.08,
  maxSteps: 300,
  replanInterval: 1,

  // UI state
  planning: false,
  playing: false,
  canPlay: false,
  statusText: 'Ready',
  currentStep: 0,
  posError: null,
  yawError: null,
  goalRewardTriggered: false,

  setParam: (key, value) => set({ [key]: value }),

  fetchDynamicsInfo: async () => {
    try {
      const resp = await fetch('/api/dynamics');
      const info: Record<string, DynamicsInfo> = await resp.json();
      set({ dynamicsInfo: info });
      // Update kwargs for current dynamics
      const state = get();
      const entry = info[state.dynamics];
      if (entry) set({ dynamicsKwargs: { ...entry.kwargs } });
    } catch (e) {
      console.warn('Could not fetch dynamics info:', e);
    }
  },

  updateDynamicsKwargs: () => {
    const { dynamics, dynamicsInfo } = get();
    const entry = dynamicsInfo[dynamics];
    if (entry) set({ dynamicsKwargs: { ...entry.kwargs } });
  },

  loadPreset: (name: string) => {
    const p = PRESETS[name];
    if (!p) return;
    set({
      backend: p.backend || 'spline',
      dynamics: p.dynamics,
      dynamicsKwargs: { ...p.kwargs },
      dt: p.dt,
      nRollouts: p.n_rollouts,
      sigma: p.sigma,
      steerSigma: p.steer_sigma || 0.05,
      lam: p.lam,
      horizonKnots: p.horizon_knots,
      numIntermediate: p.num_intermediate,
      goalPosWeight: p.goal_pos_weight,
      goalHeadingWeight: p.goal_heading_weight,
      goalVelWeight: p.goal_vel_weight || 0,
      collisionWeight: p.collision_weight,
      collisionDecay: p.collision_decay,
      collisionInsideWeight: p.collision_inside_weight,
      smoothnessWeight: p.smoothness_weight,
      terminalWeight: p.terminal_weight || 10.0,
      controlRegWeight: p.control_reg_weight || 0.1,
      signChangeWeight: p.sign_change_weight || 50.0,
      goalReachReward: 0.0,
      goalReachThreshold: 0.1,
      replanInterval: p.replan_interval || 1,
      gammaSigma: p.gamma_sigma || 0.7,
      covFloor: p.cov_floor || 0.001,
      stopThreshold: p.stop_threshold || 0.08,
      useGlobalPlanner: p.use_global_planner !== undefined ? p.use_global_planner : true,
      globalPlannerType: p.global_planner_type || 'rrt',
      goalTolerance: p.goal_tolerance,
      maxSteps: p.max_steps,
      canPlay: false,
      statusText: 'Preset loaded — hit Plan & Go',
    });
  },

  updateStepInfo: (step, posErr, yawErr, goalReward) => set({
    currentStep: step,
    posError: posErr,
    yawError: yawErr,
    goalRewardTriggered: goalReward,
  }),

  buildQueryParams: () => {
    const s = get();
    const params = new URLSearchParams({
      dynamics: s.dynamics,
      dt: String(s.dt),
      n_rollouts: String(s.nRollouts),
      sigma: String(s.sigma),
      lam: String(s.lam),
      horizon_knots: String(s.horizonKnots),
      num_intermediate: String(s.numIntermediate),
      goal_pos_weight: String(s.goalPosWeight),
      goal_heading_weight: String(s.goalHeadingWeight),
      goal_vel_weight: String(s.goalVelWeight),
      collision_weight: String(s.collisionWeight),
      collision_decay: String(s.collisionDecay),
      collision_inside_weight: String(s.collisionInsideWeight),
      smoothness_weight: String(s.smoothnessWeight),
      terminal_weight: String(s.terminalWeight),
      control_reg_weight: String(s.controlRegWeight),
      sign_change_weight: String(s.signChangeWeight),
      goal_reach_reward: String(s.goalReachReward),
      goal_reach_threshold: String(s.goalReachThreshold),
      backend: s.backend,
      gamma_sigma: String(s.gammaSigma),
      cov_floor: String(s.covFloor),
      steer_sigma: String(s.steerSigma),
      resolution: String(s.resolution),
      inflate_radius: String(s.inflateRadius),
      use_global_planner: String(s.useGlobalPlanner),
      global_planner_type: s.globalPlannerType,
      goal_tolerance: String(s.goalTolerance),
      stop_threshold: String(s.stopThreshold),
      max_steps: String(s.maxSteps),
      replan_interval: String(s.replanInterval),
    });
    // Append dynamics kwargs
    for (const [key, val] of Object.entries(s.dynamicsKwargs)) {
      params.set(key, String(val));
    }
    return params;
  },
}));
