import React, { forwardRef } from 'react';
import { Layer, Rect, Circle } from 'react-konva';
import Konva from 'konva';
import type { Obstacle, WorldBounds } from '../../../types';
import { worldToCanvas, canvasToWorld } from '../../../utils/coordinates';
import { useCanvasStore } from '../../../stores/canvasStore';

const FILL = 'rgba(59,130,246,0.3)';
const STROKE = '#3b82f6';

const ObstacleRect = React.memo(function ObstacleRect({
  obs,
  scale,
  bounds,
}: {
  obs: Obstacle;
  scale: number;
  bounds: WorldBounds;
}) {
  const updateObstaclePosition = useCanvasStore((s) => s.updateObstaclePosition);
  const removeObstacle = useCanvasStore((s) => s.removeObstacle);

  const tl = worldToCanvas(obs.x1!, obs.y2!, scale, bounds);
  const br = worldToCanvas(obs.x2!, obs.y1!, scale, bounds);
  const w = br.x - tl.x;
  const h = br.y - tl.y;

  return (
    <Rect
      x={tl.x}
      y={tl.y}
      width={w}
      height={h}
      fill={FILL}
      stroke={STROKE}
      strokeWidth={1.5}
      draggable
      onDragEnd={(e: Konva.KonvaEventObject<DragEvent>) => {
        const node = e.target;
        const newTL = canvasToWorld(node.x(), node.y(), scale, bounds);
        const newBR = canvasToWorld(node.x() + w, node.y() + h, scale, bounds);
        updateObstaclePosition(obs.id, {
          x1: Math.min(newTL.x, newBR.x),
          y1: Math.min(newTL.y, newBR.y),
          x2: Math.max(newTL.x, newBR.x),
          y2: Math.max(newTL.y, newBR.y),
        });
      }}
      onDblClick={() => removeObstacle(obs.id)}
      onDblTap={() => removeObstacle(obs.id)}
    />
  );
});

const ObstacleCircle = React.memo(function ObstacleCircle({
  obs,
  scale,
  bounds,
}: {
  obs: Obstacle;
  scale: number;
  bounds: WorldBounds;
}) {
  const updateObstaclePosition = useCanvasStore((s) => s.updateObstaclePosition);
  const removeObstacle = useCanvasStore((s) => s.removeObstacle);

  const center = worldToCanvas(obs.cx!, obs.cy!, scale, bounds);
  const canvasR = obs.r! * scale;

  return (
    <Circle
      x={center.x}
      y={center.y}
      radius={canvasR}
      fill={FILL}
      stroke={STROKE}
      strokeWidth={1.5}
      draggable
      onDragEnd={(e: Konva.KonvaEventObject<DragEvent>) => {
        const node = e.target;
        const newCenter = canvasToWorld(node.x(), node.y(), scale, bounds);
        updateObstaclePosition(obs.id, { cx: newCenter.x, cy: newCenter.y });
      }}
      onDblClick={() => removeObstacle(obs.id)}
      onDblTap={() => removeObstacle(obs.id)}
    />
  );
});

export const ObstacleLayer = forwardRef<Konva.Layer>(function ObstacleLayer(_props, ref) {
  const obstacles = useCanvasStore((s) => s.obstacles);
  const scale = useCanvasStore((s) => s.scale);
  const bounds = useCanvasStore((s) => s.worldBounds);

  return (
    <Layer ref={ref}>
      {obstacles.map((obs) =>
        obs.type === 'rect' ? (
          <ObstacleRect key={obs.id} obs={obs} scale={scale} bounds={bounds} />
        ) : (
          <ObstacleCircle key={obs.id} obs={obs} scale={scale} bounds={bounds} />
        ),
      )}
    </Layer>
  );
});
