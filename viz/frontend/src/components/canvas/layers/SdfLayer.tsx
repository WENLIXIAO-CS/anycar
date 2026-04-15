import { useEffect, useMemo, useState } from 'react';
import { Layer, Image } from 'react-konva';
import type { WorldBounds } from '../../../types';
import { worldToCanvas } from '../../../utils/coordinates';
import { renderSdfToCanvas } from '../../../utils/sdfRenderer';
import { useCanvasStore } from '../../../stores/canvasStore';

interface SdfLayerProps {
  scale: number;
  bounds: WorldBounds;
}

interface SdfImageState {
  image: HTMLCanvasElement;
  x: number;
  y: number;
  width: number;
  height: number;
}

export function SdfLayer({ scale, bounds }: SdfLayerProps) {
  const sdfData = useCanvasStore((s) => s.sdfData);
  const [state, setState] = useState<SdfImageState | null>(null);

  const placeholder = useMemo(() => {
    const c = document.createElement('canvas');
    c.width = 1;
    c.height = 1;
    return c;
  }, []);

  useEffect(() => {
    if (!sdfData) {
      setState(null);
      return;
    }

    const { grid, origin, resolution } = sdfData;
    const canvas = renderSdfToCanvas(grid, origin, resolution);
    if (!canvas) return;

    const W = grid.length;
    const H = grid[0].length;
    const worldW = W * resolution;
    const worldH = H * resolution;

    const topLeft = worldToCanvas(origin[0], origin[1] + worldH, scale, bounds);

    setState({
      image: canvas,
      x: topLeft.x,
      y: topLeft.y,
      width: worldW * scale,
      height: worldH * scale,
    });
  }, [sdfData, scale, bounds]);

  return (
    <Layer listening={false}>
      {state ? (
        <Image
          image={state.image}
          x={state.x}
          y={state.y}
          width={state.width}
          height={state.height}
        />
      ) : (
        <Image image={placeholder} visible={false} />
      )}
    </Layer>
  );
}
