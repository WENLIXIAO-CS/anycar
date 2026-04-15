import type { WorldBounds } from '../types';

export function worldToCanvas(
  wx: number, wy: number,
  scale: number, bounds: WorldBounds,
): { x: number; y: number } {
  return {
    x: (wx - bounds.minX) * scale,
    y: (bounds.maxY - wy) * scale,
  };
}

export function canvasToWorld(
  cx: number, cy: number,
  scale: number, bounds: WorldBounds,
): { x: number; y: number } {
  return {
    x: cx / scale + bounds.minX,
    y: bounds.maxY - cy / scale,
  };
}
