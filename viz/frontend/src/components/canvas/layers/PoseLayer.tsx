import { forwardRef } from 'react';
import { Layer, Group, RegularPolygon, Line } from 'react-konva';
import Konva from 'konva';
import type { Pose, WorldBounds } from '../../../types';
import { worldToCanvas } from '../../../utils/coordinates';
import { useCanvasStore } from '../../../stores/canvasStore';

function PoseMarker({
  pose,
  color,
  scale,
  bounds,
}: {
  pose: Pose;
  color: string;
  scale: number;
  bounds: WorldBounds;
}) {
  const { x, y } = worldToCanvas(pose.x, pose.y, scale, bounds);
  const radius = Math.max(8, scale * 0.08);
  const arrowLen = radius * 1.8;

  return (
    <Group x={x} y={y} rotation={-pose.heading * (180 / Math.PI)}>
      <RegularPolygon
        sides={3}
        radius={radius}
        fill={color}
        stroke="#fff"
        strokeWidth={1.5}
        rotation={-90}
      />
      <Line
        points={[0, 0, arrowLen, 0]}
        stroke="#fff"
        strokeWidth={1.5}
        lineCap="round"
      />
    </Group>
  );
}

export const PoseLayer = forwardRef<Konva.Layer>(function PoseLayer(_props, ref) {
  const startPose = useCanvasStore((s) => s.startPose);
  const goalPose = useCanvasStore((s) => s.goalPose);
  const scale = useCanvasStore((s) => s.scale);
  const bounds = useCanvasStore((s) => s.worldBounds);

  return (
    <Layer ref={ref}>
      {startPose && (
        <PoseMarker pose={startPose} color="#22c55e" scale={scale} bounds={bounds} />
      )}
      {goalPose && (
        <PoseMarker pose={goalPose} color="#ef4444" scale={scale} bounds={bounds} />
      )}
    </Layer>
  );
});
