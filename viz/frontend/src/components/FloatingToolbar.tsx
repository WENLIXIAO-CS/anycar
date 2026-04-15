import { useRef, useState, useCallback } from 'react';
import { usePlannerStore } from '../stores/plannerStore';
import { useCanvasStore } from '../stores/canvasStore';

export function FloatingToolbar() {
  const planning = usePlannerStore((s) => s.planning);
  const playing = usePlannerStore((s) => s.playing);
  const canPlay = usePlannerStore((s) => s.canPlay);
  const statusText = usePlannerStore((s) => s.statusText);
  const setParam = usePlannerStore((s) => s.setParam);
  const clearAll = useCanvasStore((s) => s.clearAll);

  const busy = planning || playing;

  // Dragging state
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const dragging = useRef(false);
  const dragOffset = useRef({ x: 0, y: 0 });
  const barRef = useRef<HTMLDivElement>(null);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    // Only drag from the handle area (not buttons)
    if ((e.target as HTMLElement).closest('button')) return;
    dragging.current = true;
    const rect = barRef.current!.getBoundingClientRect();
    dragOffset.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    e.preventDefault();
  }, []);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current) return;
    setPos({
      x: e.clientX - dragOffset.current.x,
      y: e.clientY - dragOffset.current.y,
    });
  }, []);

  const onMouseUp = useCallback(() => {
    dragging.current = false;
  }, []);

  // Attach global listeners for drag
  useState(() => {
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  });

  const style: React.CSSProperties = pos
    ? { position: 'fixed', left: pos.x, top: pos.y }
    : { position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)' };

  return (
    <div
      ref={barRef}
      onMouseDown={onMouseDown}
      style={style}
      className="z-50 flex items-center gap-3 bg-base-300 border border-base-content/10 rounded-xl shadow-2xl px-4 py-2 cursor-grab active:cursor-grabbing select-none"
    >
      <span className="text-xs opacity-50 mr-1 hidden sm:inline">{statusText}</span>

      <button
        className="btn btn-primary btn-sm"
        disabled={busy}
        onClick={() => window.dispatchEvent(new CustomEvent('start-planning'))}
      >
        {planning ? <span className="loading loading-spinner loading-xs" /> : 'Plan & Go'}
      </button>

      <button
        className="btn btn-accent btn-sm"
        disabled={!canPlay || busy}
        onClick={() => window.dispatchEvent(new CustomEvent('start-playback'))}
      >
        {playing ? <span className="loading loading-spinner loading-xs" /> : 'Play'}
      </button>

      <button
        className="btn btn-ghost btn-sm"
        onClick={() => {
          clearAll();
          setParam('planning', false);
          setParam('playing', false);
          setParam('canPlay', false);
          setParam('statusText', 'Ready');
        }}
      >
        Clear
      </button>
    </div>
  );
}
