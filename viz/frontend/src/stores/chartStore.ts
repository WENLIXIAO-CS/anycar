import { create } from 'zustand';

interface ChartState {
  distanceData: Array<{ step: number; posError: number; tolerance: number }>;
  actionData: Array<Record<string, number>>;
  actionLabels: string[];

  addDataPoint: (step: number, posError: number, action: number[]) => void;
  reset: (labels: string[]) => void;
}

export const useChartStore = create<ChartState>((set, get) => ({
  distanceData: [],
  actionData: [],
  actionLabels: [],

  addDataPoint: (step, posError, action) => {
    const { actionLabels } = get();
    const actionPoint: Record<string, number> = { step };
    actionLabels.forEach((label, i) => {
      actionPoint[label] = action[i] ?? 0;
    });

    set((s) => ({
      distanceData: [...s.distanceData, { step, posError, tolerance: 0.05 }],
      actionData: [...s.actionData, actionPoint],
    }));
  },

  reset: (labels) => set({
    distanceData: [],
    actionData: [],
    actionLabels: labels,
  }),
}));
