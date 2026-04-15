import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { Layer } from 'react-konva';
import Konva from 'konva';

interface RobotLayerProps {
  scale: number;
}

export interface RobotLayerHandle {
  show(): void;
  hide(): void;
  moveTo(canvasX: number, canvasY: number, headingRad: number): void;
}

export const RobotLayer = forwardRef<RobotLayerHandle, RobotLayerProps>(
  function RobotLayer({ scale }, ref) {
    const layerRef = useRef<Konva.Layer>(null);
    const robotRef = useRef<Konva.RegularPolygon | null>(null);

    const radius = Math.max(8, scale * 0.08);

    useEffect(() => {
      const layer = layerRef.current;
      if (!layer) return;

      if (robotRef.current) {
        robotRef.current.destroy();
      }

      const robot = new Konva.RegularPolygon({
        sides: 3,
        radius,
        fill: '#ffffff',
        stroke: '#94a3b8',
        strokeWidth: 1.5,
        visible: false,
        rotation: 0,
      });

      layer.add(robot);
      robotRef.current = robot;
      layer.batchDraw();
    }, [radius]);

    useImperativeHandle(ref, () => ({
      show() {
        const r = robotRef.current;
        if (!r) return;
        r.visible(true);
        r.getLayer()?.batchDraw();
      },
      hide() {
        const r = robotRef.current;
        if (!r) return;
        r.visible(false);
        r.getLayer()?.batchDraw();
      },
      moveTo(canvasX: number, canvasY: number, headingRad: number) {
        const r = robotRef.current;
        if (!r) return;
        r.x(canvasX);
        r.y(canvasY);
        r.rotation(-headingRad * (180 / Math.PI) + 90);
        r.getLayer()?.batchDraw();
      },
    }));

    return <Layer ref={layerRef} listening={false} />;
  },
);
