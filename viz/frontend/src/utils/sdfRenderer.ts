/**
 * Render an SDF grid to an offscreen canvas for use as a Konva.Image source.
 * Ported from planner.js renderSDF().
 */
export function renderSdfToCanvas(
  sdfGrid: number[][],
  _origin: number[],
  _resolution: number,
): HTMLCanvasElement | null {
  const W = sdfGrid.length;
  if (W === 0) return null;
  const H = sdfGrid[0].length;

  const offscreen = document.createElement('canvas');
  offscreen.width = W;
  offscreen.height = H;
  const ctx = offscreen.getContext('2d');
  if (!ctx) return null;

  const imgData = ctx.createImageData(W, H);

  for (let i = 0; i < W; i++) {
    for (let j = 0; j < H; j++) {
      const val = sdfGrid[i][j];
      // Flip j for canvas Y direction
      const pixIdx = ((H - 1 - j) * W + i) * 4;

      if (val < 0) {
        // Inside obstacle: red
        imgData.data[pixIdx] = 220;
        imgData.data[pixIdx + 1] = 50;
        imgData.data[pixIdx + 2] = 50;
        imgData.data[pixIdx + 3] = 180;
      } else if (val < 0.2) {
        // Near obstacle surface: yellow/orange fade
        const t = val / 0.2;
        imgData.data[pixIdx] = 220;
        imgData.data[pixIdx + 1] = Math.floor(50 + 170 * t);
        imgData.data[pixIdx + 2] = 50;
        imgData.data[pixIdx + 3] = Math.floor(180 * (1 - t * 0.7));
      } else {
        // Free space: transparent
        imgData.data[pixIdx] = 0;
        imgData.data[pixIdx + 1] = 0;
        imgData.data[pixIdx + 2] = 0;
        imgData.data[pixIdx + 3] = 0;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return offscreen;
}
