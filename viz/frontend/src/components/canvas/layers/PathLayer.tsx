import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { Layer } from 'react-konva';
import Konva from 'konva';
import type { WorldBounds } from '../../../types';
import { worldToCanvas } from '../../../utils/coordinates';

const NUM_SAMPLES = 10;

const SAMPLE_COLORS = [
  '#22c55e',
  '#4ade50',
  '#84cc16',
  '#a3e635',
  '#facc15',
  '#fbbf24',
  '#f97316',
  '#ef4444',
  '#dc2626',
  '#b91c1c',
];

export interface PathLayerHandle {
  setGlobalPath(path: number[][], scale: number, bounds: WorldBounds): void;
  setTrajectoryPoints(points: { x: number; y: number }[]): void;
  setHorizonPoints(points: { x: number; y: number }[]): void;
  setSampleLines(samples: { x: number; y: number }[][]): void;
  setSmoothedPath(points: { x: number; y: number }[]): void;
  clear(): void;
}

export const PathLayer = forwardRef<PathLayerHandle>(function PathLayer(_props, ref) {
  const layerRef = useRef<Konva.Layer>(null);
  const globalPathRef = useRef<Konva.Line | null>(null);
  const trajectoryRef = useRef<Konva.Line | null>(null);
  const horizonRef = useRef<Konva.Line | null>(null);
  const sampleRefs = useRef<Konva.Line[]>([]);
  const smoothedRef = useRef<Konva.Line | null>(null);
  const arrowsGroupRef = useRef<Konva.Group | null>(null);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;

    const globalPath = new Konva.Line({
      points: [],
      stroke: '#22c55e',
      strokeWidth: 2,
      dash: [8, 4],
      lineCap: 'round',
      lineJoin: 'round',
    });

    const arrowsGroup = new Konva.Group();

    const trajectory = new Konva.Line({
      points: [],
      stroke: '#38bdf8',
      strokeWidth: 2.5,
      lineCap: 'round',
      lineJoin: 'round',
    });

    const horizon = new Konva.Line({
      points: [],
      stroke: '#06b6d4',
      strokeWidth: 1.5,
      dash: [6, 4],
      opacity: 0.6,
      lineCap: 'round',
      lineJoin: 'round',
    });

    const samples: Konva.Line[] = [];
    for (let i = 0; i < NUM_SAMPLES; i++) {
      const line = new Konva.Line({
        points: [],
        stroke: SAMPLE_COLORS[i],
        strokeWidth: 1,
        opacity: 0.5,
        lineCap: 'round',
        lineJoin: 'round',
      });
      samples.push(line);
      layer.add(line);
    }

    const smoothed = new Konva.Line({
      points: [],
      stroke: '#f97316',
      strokeWidth: 2,
      dash: [6, 4],
      lineCap: 'round',
      lineJoin: 'round',
    });

    layer.add(globalPath);
    layer.add(arrowsGroup);
    layer.add(trajectory);
    layer.add(horizon);
    layer.add(smoothed);

    globalPathRef.current = globalPath;
    arrowsGroupRef.current = arrowsGroup;
    trajectoryRef.current = trajectory;
    horizonRef.current = horizon;
    sampleRefs.current = samples;
    smoothedRef.current = smoothed;

    layer.batchDraw();
  }, []);

  useImperativeHandle(ref, () => ({
    setGlobalPath(path: number[][], scale: number, bounds: WorldBounds) {
      const line = globalPathRef.current;
      const group = arrowsGroupRef.current;
      if (!line || !group) return;

      const flat: number[] = [];
      group.destroyChildren();

      for (let i = 0; i < path.length; i++) {
        const { x, y } = worldToCanvas(path[i][0], path[i][1], scale, bounds);
        flat.push(x, y);

        if (i > 0 && i % 3 === 0) {
          const prev = worldToCanvas(path[i - 1][0], path[i - 1][1], scale, bounds);
          const arrow = new Konva.Arrow({
            points: [prev.x, prev.y, x, y],
            pointerLength: 6,
            pointerWidth: 4,
            fill: '#22c55e',
            stroke: '#22c55e',
            strokeWidth: 1,
            opacity: 0.6,
          });
          group.add(arrow);
        }
      }

      line.points(flat);
      line.getLayer()?.batchDraw();
    },

    setTrajectoryPoints(points: { x: number; y: number }[]) {
      const line = trajectoryRef.current;
      if (!line) return;
      line.points(points.flatMap((p) => [p.x, p.y]));
      line.getLayer()?.batchDraw();
    },

    setHorizonPoints(points: { x: number; y: number }[]) {
      const line = horizonRef.current;
      if (!line) return;
      line.points(points.flatMap((p) => [p.x, p.y]));
      line.getLayer()?.batchDraw();
    },

    setSampleLines(samples: { x: number; y: number }[][]) {
      const lines = sampleRefs.current;
      for (let i = 0; i < NUM_SAMPLES; i++) {
        if (i < samples.length) {
          lines[i].points(samples[i].flatMap((p) => [p.x, p.y]));
          lines[i].visible(true);
        } else {
          lines[i].points([]);
          lines[i].visible(false);
        }
      }
      lines[0]?.getLayer()?.batchDraw();
    },

    setSmoothedPath(points: { x: number; y: number }[]) {
      const line = smoothedRef.current;
      if (!line) return;
      line.points(points.flatMap((p) => [p.x, p.y]));
      line.getLayer()?.batchDraw();
    },

    clear() {
      globalPathRef.current?.points([]);
      arrowsGroupRef.current?.destroyChildren();
      trajectoryRef.current?.points([]);
      horizonRef.current?.points([]);
      smoothedRef.current?.points([]);
      for (const s of sampleRefs.current) {
        s.points([]);
        s.visible(false);
      }
      layerRef.current?.batchDraw();
    },
  }));

  return <Layer ref={layerRef} listening={false} />;
});
