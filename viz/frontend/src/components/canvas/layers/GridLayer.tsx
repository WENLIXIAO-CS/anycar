import React from 'react';
import { Layer, Line, Text } from 'react-konva';
import type { WorldBounds } from '../../../types';
import { worldToCanvas } from '../../../utils/coordinates';

interface GridLayerProps {
  scale: number;
  bounds: WorldBounds;
}

export const GridLayer = React.memo(function GridLayer({ scale, bounds }: GridLayerProps) {
  const lines: React.ReactNode[] = [];
  const labels: React.ReactNode[] = [];
  const fontSize = Math.max(9, Math.round(scale * 0.1));
  const step = 0.5;

  for (let wx = Math.ceil(bounds.minX / step) * step; wx <= bounds.maxX; wx = +(wx + step).toFixed(4)) {
    const { x } = worldToCanvas(wx, 0, scale, bounds);
    const isAxis = Math.abs(wx) < 1e-9;
    lines.push(
      <Line
        key={`gv-${wx}`}
        points={[x, 0, x, (bounds.maxY - bounds.minY) * scale]}
        stroke={isAxis ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.1)'}
        strokeWidth={isAxis ? 1.5 : 0.5}
      />,
    );
    if (Math.abs(wx - Math.round(wx)) < 1e-9) {
      const { y: labelY } = worldToCanvas(wx, 0, scale, bounds);
      labels.push(
        <Text
          key={`lx-${wx}`}
          x={x + 2}
          y={labelY + 2}
          text={String(Math.round(wx))}
          fontSize={fontSize}
          fill="rgba(255,255,255,0.4)"
        />,
      );
    }
  }

  for (let wy = Math.ceil(bounds.minY / step) * step; wy <= bounds.maxY; wy = +(wy + step).toFixed(4)) {
    const { y } = worldToCanvas(0, wy, scale, bounds);
    const isAxis = Math.abs(wy) < 1e-9;
    lines.push(
      <Line
        key={`gh-${wy}`}
        points={[0, y, (bounds.maxX - bounds.minX) * scale, y]}
        stroke={isAxis ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.1)'}
        strokeWidth={isAxis ? 1.5 : 0.5}
      />,
    );
    if (Math.abs(wy - Math.round(wy)) < 1e-9) {
      const { x: labelX } = worldToCanvas(0, wy, scale, bounds);
      labels.push(
        <Text
          key={`ly-${wy}`}
          x={labelX + 2}
          y={y + 2}
          text={String(Math.round(wy))}
          fontSize={fontSize}
          fill="rgba(255,255,255,0.4)"
        />,
      );
    }
  }

  return (
    <Layer listening={false}>
      {lines}
      {labels}
    </Layer>
  );
});
