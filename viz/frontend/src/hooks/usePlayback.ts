import { useRef, useCallback, useEffect } from 'react';
import { usePlannerStore } from '../stores/plannerStore';
import { useCanvasStore } from '../stores/canvasStore';
import { useChartStore } from '../stores/chartStore';
import { ACTION_LABELS } from '../constants/dynamics';
import { worldToCanvas } from '../utils/coordinates';
import type { PathLayerHandle } from '../components/canvas/layers/PathLayer';
import type { RobotLayerHandle } from '../components/canvas/layers/RobotLayer';

export function usePlayback(
  pathLayerRef: React.RefObject<PathLayerHandle | null>,
  robotLayerRef: React.RefObject<RobotLayerHandle | null>,
): {
  startPlayback: () => void;
} {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      clearTimer();
    };
  }, [clearTimer]);

  const startPlayback = useCallback(() => {
    const cs = useCanvasStore.getState();
    const steps = cs.storedSteps;

    if (steps.length === 0) return;

    clearTimer();

    const ps = usePlannerStore.getState();
    const ch = useChartStore.getState();

    // Set planner state
    ps.setParam('playing', true);
    ps.setParam('statusText', 'Playing...');
    ps.setParam('currentStep', 0);
    ps.setParam('posError', null);
    ps.setParam('yawError', null);

    // Reset charts
    ch.reset(ACTION_LABELS[ps.dynamics] || ['a0', 'a1', 'a2']);

    // Clear path layer, show robot
    pathLayerRef.current?.clear();
    robotLayerRef.current?.show();

    // Get scale and bounds
    const scale = cs.scale;
    const bounds = cs.worldBounds;
    const smoothedTrajectory = cs.smoothedTrajectory;

    // Accumulated trajectory points for the blue line
    const trajectoryPoints: { x: number; y: number }[] = [];

    let i = 0;

    function tick() {
      if (i >= steps.length) {
        // Playback complete
        pathLayerRef.current?.setHorizonPoints([]);
        pathLayerRef.current?.setSampleLines([]);

        if (smoothedTrajectory && smoothedTrajectory.length > 0) {
          const smoothedPts = smoothedTrajectory.map((pt) =>
            worldToCanvas(pt[0], pt[1], scale, bounds),
          );
          pathLayerRef.current?.setSmoothedPath(smoothedPts);
        }

        const pState = usePlannerStore.getState();
        pState.setParam('playing', false);
        pState.setParam('statusText', 'Done');

        window.dispatchEvent(new CustomEvent('playback-done'));
        timerRef.current = null;
        return;
      }

      const step = steps[i];

      // Convert state position to canvas coords and move robot
      const pos = worldToCanvas(step.state[0], step.state[1], scale, bounds);
      robotLayerRef.current?.moveTo(pos.x, pos.y, step.state[2]);

      // Accumulate trajectory and update line
      trajectoryPoints.push(pos);
      pathLayerRef.current?.setTrajectoryPoints(trajectoryPoints);

      // Update predicted trajectory (horizon)
      if (step.predicted_trajectory && step.predicted_trajectory.length > 0) {
        const horizonPts = step.predicted_trajectory.map((pt) =>
          worldToCanvas(pt[0], pt[1], scale, bounds),
        );
        pathLayerRef.current?.setHorizonPoints(horizonPts);
      }

      // Update sampled trajectories
      if (step.sampled_trajectories && step.sampled_trajectories.length > 0) {
        const sampleLines = step.sampled_trajectories.map((traj) =>
          traj.map((pt) => worldToCanvas(pt[0], pt[1], scale, bounds)),
        );
        pathLayerRef.current?.setSampleLines(sampleLines);
      }

      // Update stores
      usePlannerStore.getState().updateStepInfo(
        step.step,
        step.pos_error,
        step.heading_error,
        step.goal_reward_triggered,
      );
      useChartStore.getState().addDataPoint(step.step, step.pos_error, step.action);

      i++;
      timerRef.current = setTimeout(tick, 30);
    }

    tick();
  }, [pathLayerRef, robotLayerRef, clearTimer]);

  return { startPlayback };
}
