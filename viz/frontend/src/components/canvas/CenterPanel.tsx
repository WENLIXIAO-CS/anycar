import { useRef, useEffect } from 'react';
import { KonvaCanvas } from './KonvaCanvas';
import { useSSEPlanning } from '../../hooks/useSSEPlanning';
import { usePlayback } from '../../hooks/usePlayback';
import type { PathLayerHandle } from './layers/PathLayer';
import type { RobotLayerHandle } from './layers/RobotLayer';

export function CenterPanel() {
  const containerRef = useRef<HTMLDivElement>(null);
  const pathLayerRef = useRef<PathLayerHandle>(null);
  const robotLayerRef = useRef<RobotLayerHandle>(null);

  const { startPlanning } = useSSEPlanning();
  const { startPlayback } = usePlayback(pathLayerRef, robotLayerRef);

  // Listen for planning/playback events from ActionButtons
  useEffect(() => {
    const onPlan = () => startPlanning();
    const onPlay = () => startPlayback();
    window.addEventListener('start-planning', onPlan);
    window.addEventListener('start-playback', onPlay);
    return () => {
      window.removeEventListener('start-planning', onPlan);
      window.removeEventListener('start-playback', onPlay);
    };
  }, [startPlanning, startPlayback]);

  return (
    <section className="flex-1 flex flex-col p-4 min-w-0 overflow-hidden">
      <div
        ref={containerRef}
        id="konva-container"
        className="w-full flex-1 bg-base-300 rounded-lg"
      >
        <KonvaCanvas
          containerRef={containerRef}
          pathLayerRef={pathLayerRef}
          robotLayerRef={robotLayerRef}
        />
      </div>
    </section>
  );
}
