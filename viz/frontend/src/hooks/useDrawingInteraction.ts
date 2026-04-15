import { useEffect, useRef } from 'react';
import Konva from 'konva';
import { useCanvasStore, genObstacleId } from '../stores/canvasStore';
import type { CanvasState } from '../stores/canvasStore';
import { canvasToWorld } from '../utils/coordinates';

/**
 * Attaches mouse/touch handlers on the Konva Stage for drawing obstacles
 * and placing start/goal poses.
 */
export function useDrawingInteraction(
  stageRef: React.RefObject<Konva.Stage | null>,
  obstacleLayerRef: React.RefObject<Konva.Layer | null>,
  poseLayerRef: React.RefObject<Konva.Layer | null>,
): void {
  const isDrawing = useRef(false);
  const drawOrigin = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const previewShape = useRef<Konva.Rect | Konva.Circle | null>(null);
  const headingLine = useRef<Konva.Line | null>(null);

  // Keep drawMode in a ref so event handlers always see the latest value
  const drawModeRef = useRef<CanvasState['drawMode']>('rect');
  useEffect(() => {
    const unsub = useCanvasStore.subscribe(
      (s) => { drawModeRef.current = s.drawMode; },
    );
    return unsub;
  }, []);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;

    const getPointer = (): { x: number; y: number } => {
      const pos = stage.getPointerPosition();
      return pos ?? { x: 0, y: 0 };
    };

    const handleMouseDown = () => {
      const mode = drawModeRef.current;
      const pos = getPointer();
      const { removeObstacle, obstacles } = useCanvasStore.getState();

      // Erase mode: find clicked obstacle shape and remove it
      if (mode === 'erase') {
        const target = stage.getIntersection(pos);
        if (target && target.name() === 'obstacle') {
          const shapeId = target.id();
          const match = obstacles.find((o) => o.id === shapeId);
          if (match) {
            removeObstacle(match.id);
          }
        }
        return;
      }

      // If user clicked on an existing obstacle shape, let Konva drag handle it
      const target = stage.getIntersection(pos);
      if (target && target.name() === 'obstacle') {
        return;
      }

      isDrawing.current = true;
      drawOrigin.current = { x: pos.x, y: pos.y };

      const layer = (mode === 'start' || mode === 'goal')
        ? poseLayerRef.current
        : obstacleLayerRef.current;
      if (!layer) return;

      if (mode === 'rect') {
        const rect = new Konva.Rect({
          x: pos.x,
          y: pos.y,
          width: 0,
          height: 0,
          stroke: 'rgba(255,100,100,0.8)',
          strokeWidth: 2,
          dash: [6, 3],
          listening: false,
        });
        layer.add(rect);
        previewShape.current = rect;
      } else if (mode === 'circle') {
        const circle = new Konva.Circle({
          x: pos.x,
          y: pos.y,
          radius: 0,
          stroke: 'rgba(255,100,100,0.8)',
          strokeWidth: 2,
          dash: [6, 3],
          listening: false,
        });
        layer.add(circle);
        previewShape.current = circle;
      } else if (mode === 'start' || mode === 'goal') {
        const color = mode === 'start' ? '#00ff88' : '#ff4444';
        const line = new Konva.Line({
          points: [pos.x, pos.y, pos.x, pos.y],
          stroke: color,
          strokeWidth: 2,
          dash: [6, 3],
          listening: false,
        });
        layer.add(line);
        headingLine.current = line;
      }

      layer.batchDraw();
    };

    const handleMouseMove = () => {
      if (!isDrawing.current) return;
      const pos = getPointer();
      const mode = drawModeRef.current;

      if (mode === 'rect' && previewShape.current instanceof Konva.Rect) {
        const ox = drawOrigin.current.x;
        const oy = drawOrigin.current.y;
        const x = Math.min(ox, pos.x);
        const y = Math.min(oy, pos.y);
        const w = Math.abs(pos.x - ox);
        const h = Math.abs(pos.y - oy);
        previewShape.current.x(x);
        previewShape.current.y(y);
        previewShape.current.width(w);
        previewShape.current.height(h);
        previewShape.current.getLayer()?.batchDraw();
      } else if (mode === 'circle' && previewShape.current instanceof Konva.Circle) {
        const dx = pos.x - drawOrigin.current.x;
        const dy = pos.y - drawOrigin.current.y;
        const r = Math.sqrt(dx * dx + dy * dy);
        previewShape.current.radius(r);
        previewShape.current.getLayer()?.batchDraw();
      } else if ((mode === 'start' || mode === 'goal') && headingLine.current) {
        headingLine.current.points([
          drawOrigin.current.x, drawOrigin.current.y,
          pos.x, pos.y,
        ]);
        headingLine.current.getLayer()?.batchDraw();
      }
    };

    const handleMouseUp = () => {
      if (!isDrawing.current) return;
      isDrawing.current = false;

      const mode = drawModeRef.current;
      const pos = getPointer();
      const { scale, worldBounds, addObstacle, setStartPose, setGoalPose } = useCanvasStore.getState();

      if (mode === 'rect' && previewShape.current instanceof Konva.Rect) {
        const shape = previewShape.current;
        const w = shape.width();
        const h = shape.height();

        shape.destroy();
        previewShape.current = null;

        // Skip tiny shapes
        if (w / scale < 0.1 && h / scale < 0.1) return;

        const topLeft = canvasToWorld(shape.x(), shape.y(), scale, worldBounds);
        const bottomRight = canvasToWorld(shape.x() + w, shape.y() + h, scale, worldBounds);

        addObstacle({
          id: genObstacleId(),
          type: 'rect',
          x1: Math.min(topLeft.x, bottomRight.x),
          y1: Math.min(topLeft.y, bottomRight.y),
          x2: Math.max(topLeft.x, bottomRight.x),
          y2: Math.max(topLeft.y, bottomRight.y),
        });
      } else if (mode === 'circle' && previewShape.current instanceof Konva.Circle) {
        const shape = previewShape.current;
        const r = shape.radius();

        shape.destroy();
        previewShape.current = null;

        // Skip tiny shapes
        if (r / scale < 0.05) return;

        const center = canvasToWorld(shape.x(), shape.y(), scale, worldBounds);
        addObstacle({
          id: genObstacleId(),
          type: 'circle',
          cx: center.x,
          cy: center.y,
          r: r / scale,
        });
      } else if ((mode === 'start' || mode === 'goal') && headingLine.current) {
        headingLine.current.destroy();
        headingLine.current = null;

        const dx = pos.x - drawOrigin.current.x;
        const dy = pos.y - drawOrigin.current.y;
        // Canvas Y is flipped relative to world Y, so negate dy
        const heading = Math.atan2(-dy, dx);

        const worldPos = canvasToWorld(drawOrigin.current.x, drawOrigin.current.y, scale, worldBounds);

        const pose = { x: worldPos.x, y: worldPos.y, heading };
        if (mode === 'start') {
          setStartPose(pose);
        } else {
          setGoalPose(pose);
        }
      }

      // Redraw layers
      obstacleLayerRef.current?.batchDraw();
      poseLayerRef.current?.batchDraw();
    };

    stage.on('mousedown touchstart', handleMouseDown);
    stage.on('mousemove touchmove', handleMouseMove);
    stage.on('mouseup touchend', handleMouseUp);

    return () => {
      stage.off('mousedown touchstart', handleMouseDown);
      stage.off('mousemove touchmove', handleMouseMove);
      stage.off('mouseup touchend', handleMouseUp);
    };
  }, [stageRef, obstacleLayerRef, poseLayerRef]);
}
