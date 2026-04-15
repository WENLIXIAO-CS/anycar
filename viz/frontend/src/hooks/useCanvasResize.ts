import { useEffect } from 'react';
import type Konva from 'konva';
import { useCanvasStore } from '../stores/canvasStore';

/**
 * Observes a container div and keeps the Konva stage sized to fit,
 * updating the canvas scale in the store.
 */
export function useCanvasResize(
  containerRef: React.RefObject<HTMLDivElement | null>,
  stageRef: React.RefObject<Konva.Stage | null>,
): void {
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const update = () => {
      const { worldBounds, setScale } = useCanvasStore.getState();
      const worldSpan = worldBounds.maxX - worldBounds.minX;
      const size = Math.min(container.clientWidth, container.clientHeight);
      const scale = size / worldSpan;

      setScale(scale);

      const stage = stageRef.current;
      if (stage) {
        stage.width(size);
        stage.height(size);
      }
    };

    // Initial sizing
    update();

    const observer = new ResizeObserver(() => {
      update();
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
    };
  }, [containerRef, stageRef]);
}
