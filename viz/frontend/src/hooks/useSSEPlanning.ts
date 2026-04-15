import { useRef, useCallback } from 'react';
import { usePlannerStore } from '../stores/plannerStore';
import { useCanvasStore } from '../stores/canvasStore';
import { useChartStore } from '../stores/chartStore';
import { ACTION_LABELS } from '../constants/dynamics';
import type { StepData, SdfData } from '../types';

export function useSSEPlanning(): {
  startPlanning: () => void;
  abort: () => void;
} {
  const esRef = useRef<EventSource | null>(null);

  const closeES = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
  }, []);

  const abort = useCallback(() => {
    closeES();
    const ps = usePlannerStore.getState();
    ps.setParam('planning', false);
    ps.setParam('statusText', 'Aborted');
  }, [closeES]);

  const startPlanning = useCallback(() => {
    // 1. Close any existing EventSource
    closeES();

    const ps = usePlannerStore.getState();
    const cs = useCanvasStore.getState();
    const ch = useChartStore.getState();

    // 2. Clear previous planning data
    cs.clearPlanningData();

    // 3. Set planner state
    ps.setParam('planning', true);
    ps.setParam('canPlay', false);
    ps.setParam('statusText', 'Planning...');
    ps.setParam('currentStep', 0);
    ps.setParam('posError', null);
    ps.setParam('yawError', null);

    // 4. Reset charts with action labels for current dynamics
    const dynamics = ps.dynamics;
    ch.reset(ACTION_LABELS[dynamics] || ['a0', 'a1', 'a2']);

    // 5. Build URL
    const params = ps.buildQueryParams();
    const obstacles = cs.getObstaclesForApi();
    const poses = cs.getPosesForApi();
    const wb = cs.worldBounds;

    params.set('boxes', JSON.stringify(obstacles.boxes));
    params.set('circles', JSON.stringify(obstacles.circles));
    params.set('start', JSON.stringify(poses.start || [0, 0, 0]));
    params.set('goal', JSON.stringify(poses.goal || [2, 2, 0]));
    params.set('bounds', JSON.stringify([wb.minX, wb.minY, wb.maxX, wb.maxY]));

    const url = '/api/plan?' + params.toString();

    // 6. Create EventSource
    const es = new EventSource(url);
    esRef.current = es;

    // 7. Listen for events
    es.addEventListener('sdf', (e: MessageEvent) => {
      const data: SdfData = JSON.parse(e.data);
      useCanvasStore.getState().setSdfData(data);
    });

    es.addEventListener('global_path', (e: MessageEvent) => {
      const data: { path: number[][] } = JSON.parse(e.data);
      useCanvasStore.getState().setGlobalPath(data.path);
    });

    es.addEventListener('step', (e: MessageEvent) => {
      const data: StepData = JSON.parse(e.data);
      useCanvasStore.getState().appendStep(data);
    });

    es.addEventListener('done', (e: MessageEvent) => {
      const data = JSON.parse(e.data);
      closeES();

      const store = useCanvasStore.getState();
      const totalSteps = store.storedSteps.length;

      if (data.smoothed_trajectory) {
        store.setSmoothedTrajectory(data.smoothed_trajectory);
      }
      if (data.trajectory) {
        store.setPlannedTrajectory(data.trajectory);
      }

      const pState = usePlannerStore.getState();
      pState.setParam('planning', false);
      pState.setParam('canPlay', true);
      pState.setParam(
        'statusText',
        `Planned (${totalSteps} steps) \u2014 hit Play`,
      );
    });

    es.addEventListener('error', (e: Event) => {
      // SSE named 'error' event from server (MessageEvent)
      if (e instanceof MessageEvent) {
        const data = JSON.parse(e.data);
        closeES();
        const pState = usePlannerStore.getState();
        pState.setParam('planning', false);
        pState.setParam('statusText', `Error: ${data.message || 'Unknown'}`);
        return;
      }
      // Connection-level error
      closeES();
      const pState = usePlannerStore.getState();
      pState.setParam('planning', false);
      pState.setParam('statusText', 'Error: connection lost');
    });

    es.onerror = () => {
      closeES();
      const pState = usePlannerStore.getState();
      pState.setParam('planning', false);
      pState.setParam('statusText', 'Error: connection lost');
    };
  }, [closeES]);

  return { startPlanning, abort };
}
