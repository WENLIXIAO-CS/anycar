import { useEffect, useRef } from 'react';
import { Stage } from 'react-konva';
import Konva from 'konva';
import { useCanvasStore, genObstacleId } from '../../stores/canvasStore';
import { worldToCanvas } from '../../utils/coordinates';
import { useCanvasResize } from '../../hooks/useCanvasResize';
import { useDrawingInteraction } from '../../hooks/useDrawingInteraction';
import { GridLayer } from './layers/GridLayer';
import { SdfLayer } from './layers/SdfLayer';
import { ObstacleLayer } from './layers/ObstacleLayer';
import { PathLayer } from './layers/PathLayer';
import type { PathLayerHandle } from './layers/PathLayer';
import { PoseLayer } from './layers/PoseLayer';
import { RobotLayer } from './layers/RobotLayer';
import type { RobotLayerHandle } from './layers/RobotLayer';

interface KonvaCanvasProps {
  containerRef: React.RefObject<HTMLDivElement | null>;
  pathLayerRef: React.RefObject<PathLayerHandle | null>;
  robotLayerRef: React.RefObject<RobotLayerHandle | null>;
}

export function KonvaCanvas({ containerRef, pathLayerRef, robotLayerRef }: KonvaCanvasProps) {
  const stageRef = useRef<Konva.Stage>(null);
  const obstacleLayerRef = useRef<Konva.Layer>(null);
  const poseLayerRef = useRef<Konva.Layer>(null);

  const scale = useCanvasStore((s) => s.scale);
  const bounds = useCanvasStore((s) => s.worldBounds);
  const size = Math.round((bounds.maxX - bounds.minX) * scale);

  useCanvasResize(containerRef, stageRef);
  useDrawingInteraction(stageRef, obstacleLayerRef, poseLayerRef);

  // Load preset scene
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as {
        boxes: number[][];
        circles: number[][];
        start: number[];
        goal: number[];
      };
      const store = useCanvasStore.getState();
      store.clearAll();
      pathLayerRef.current?.clear();
      robotLayerRef.current?.hide();

      for (const box of detail.boxes) {
        store.addObstacle({
          id: genObstacleId(),
          type: 'rect',
          x1: Math.min(box[0], box[2]),
          y1: Math.min(box[1], box[3]),
          x2: Math.max(box[0], box[2]),
          y2: Math.max(box[1], box[3]),
        });
      }
      for (const circle of detail.circles) {
        store.addObstacle({
          id: genObstacleId(),
          type: 'circle',
          cx: circle[0],
          cy: circle[1],
          r: circle[2],
        });
      }
      if (detail.start) {
        store.setStartPose({ x: detail.start[0], y: detail.start[1], heading: detail.start[2] });
      }
      if (detail.goal) {
        store.setGoalPose({ x: detail.goal[0], y: detail.goal[1], heading: detail.goal[2] });
      }
    };
    window.addEventListener('load-preset-scene', handler);
    return () => window.removeEventListener('load-preset-scene', handler);
  }, [pathLayerRef, robotLayerRef]);

  // Sync globalPath to PathLayer
  const globalPath = useCanvasStore((s) => s.globalPath);
  useEffect(() => {
    if (globalPath && pathLayerRef.current) {
      pathLayerRef.current.setGlobalPath(globalPath, scale, bounds);
    }
  }, [globalPath, scale, bounds, pathLayerRef]);

  // Sync smoothedTrajectory to PathLayer
  const smoothedTrajectory = useCanvasStore((s) => s.smoothedTrajectory);
  useEffect(() => {
    if (smoothedTrajectory && pathLayerRef.current) {
      const points = smoothedTrajectory.map(([wx, wy]) => worldToCanvas(wx, wy, scale, bounds));
      pathLayerRef.current.setSmoothedPath(points);
    }
  }, [smoothedTrajectory, scale, bounds, pathLayerRef]);

  // Show planned trajectory immediately after planning
  const plannedTrajectory = useCanvasStore((s) => s.plannedTrajectory);
  useEffect(() => {
    if (plannedTrajectory && plannedTrajectory.length > 1 && pathLayerRef.current) {
      const points = plannedTrajectory.map(([wx, wy]) => worldToCanvas(wx, wy, scale, bounds));
      pathLayerRef.current.setTrajectoryPoints(points);
    }
  }, [plannedTrajectory, scale, bounds, pathLayerRef]);

  return (
    <Stage ref={stageRef} width={size} height={size}>
      <GridLayer scale={scale} bounds={bounds} />
      <SdfLayer scale={scale} bounds={bounds} />
      <ObstacleLayer ref={obstacleLayerRef} />
      <PathLayer ref={pathLayerRef} />
      <PoseLayer ref={poseLayerRef} />
      <RobotLayer ref={robotLayerRef} scale={scale} />
    </Stage>
  );
}
