import { create } from 'zustand';
import type { Obstacle, Pose, StepData, SdfData, WorldBounds } from '../types';

let nextId = 0;
export function genObstacleId(): string {
  return `obs-${nextId++}`;
}

export interface CanvasState {
  worldBounds: WorldBounds;
  scale: number;
  drawMode: 'rect' | 'circle' | 'start' | 'goal' | 'erase';

  obstacles: Obstacle[];
  startPose: Pose | null;
  goalPose: Pose | null;

  sdfData: SdfData | null;
  globalPath: number[][] | null;
  storedSteps: StepData[];
  smoothedTrajectory: number[][] | null;
  plannedTrajectory: number[][] | null;

  setScale: (scale: number) => void;
  setDrawMode: (mode: CanvasState['drawMode']) => void;
  addObstacle: (obs: Obstacle) => void;
  removeObstacle: (id: string) => void;
  updateObstaclePosition: (id: string, coords: Partial<Obstacle>) => void;
  setStartPose: (pose: Pose) => void;
  setGoalPose: (pose: Pose) => void;
  setSdfData: (data: SdfData) => void;
  setGlobalPath: (path: number[][]) => void;
  appendStep: (step: StepData) => void;
  setSmoothedTrajectory: (traj: number[][]) => void;
  setPlannedTrajectory: (traj: number[][]) => void;
  clearAll: () => void;
  clearPlanningData: () => void;
  setObstacles: (obs: Obstacle[]) => void;

  getObstaclesForApi: () => { boxes: number[][]; circles: number[][] };
  getPosesForApi: () => { start: number[] | null; goal: number[] | null };
}

export const useCanvasStore = create<CanvasState>((set, get) => ({
  worldBounds: { minX: -3, maxX: 3, minY: -3, maxY: 3 },
  scale: 100,
  drawMode: 'rect',

  obstacles: [],
  startPose: null,
  goalPose: null,

  sdfData: null,
  globalPath: null,
  storedSteps: [],
  smoothedTrajectory: null,
  plannedTrajectory: null,

  setScale: (scale) => set({ scale }),
  setDrawMode: (mode) => set({ drawMode: mode }),
  addObstacle: (obs) => set((s) => ({ obstacles: [...s.obstacles, obs] })),
  removeObstacle: (id) => set((s) => ({
    obstacles: s.obstacles.filter((o) => o.id !== id),
  })),
  updateObstaclePosition: (id, coords) => set((s) => ({
    obstacles: s.obstacles.map((o) => o.id === id ? { ...o, ...coords } : o),
  })),
  setStartPose: (pose) => set({ startPose: pose }),
  setGoalPose: (pose) => set({ goalPose: pose }),
  setSdfData: (data) => set({ sdfData: data }),
  setGlobalPath: (path) => set({ globalPath: path }),
  appendStep: (step) => set((s) => ({ storedSteps: [...s.storedSteps, step] })),
  setSmoothedTrajectory: (traj) => set({ smoothedTrajectory: traj }),
  setPlannedTrajectory: (traj) => set({ plannedTrajectory: traj }),
  setObstacles: (obs) => set({ obstacles: obs }),
  clearPlanningData: () => set({
    sdfData: null,
    globalPath: null,
    storedSteps: [],
    smoothedTrajectory: null,
    plannedTrajectory: null,
  }),
  clearAll: () => set({
    obstacles: [],
    startPose: null,
    goalPose: null,
    sdfData: null,
    globalPath: null,
    storedSteps: [],
    smoothedTrajectory: null,
    plannedTrajectory: null,
  }),

  getObstaclesForApi: () => {
    const { obstacles } = get();
    const boxes: number[][] = [];
    const circles: number[][] = [];
    for (const o of obstacles) {
      if (o.type === 'rect') {
        boxes.push([o.x1!, o.y1!, o.x2!, o.y2!]);
      } else {
        circles.push([o.cx!, o.cy!, o.r!]);
      }
    }
    return { boxes, circles };
  },

  getPosesForApi: () => {
    const { startPose, goalPose } = get();
    return {
      start: startPose ? [startPose.x, startPose.y, startPose.heading] : null,
      goal: goalPose ? [goalPose.x, goalPose.y, goalPose.heading] : null,
    };
  },
}));
