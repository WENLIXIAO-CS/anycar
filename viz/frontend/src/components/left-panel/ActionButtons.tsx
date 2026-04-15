import { usePlannerStore } from '../../stores/plannerStore';
import { useCanvasStore } from '../../stores/canvasStore';

export function ActionButtons() {
  const planning = usePlannerStore((s) => s.planning);
  const playing = usePlannerStore((s) => s.playing);
  const canPlay = usePlannerStore((s) => s.canPlay);
  const setParam = usePlannerStore((s) => s.setParam);
  const clearAll = useCanvasStore((s) => s.clearAll);

  const busy = planning || playing;

  return (
    <div className="flex flex-col gap-2">
      <button
        className="btn btn-primary btn-sm w-full"
        disabled={busy}
        onClick={() => window.dispatchEvent(new CustomEvent('start-planning'))}
      >
        Plan &amp; Go
      </button>

      {canPlay && (
        <button
          className="btn btn-accent btn-sm w-full"
          disabled={busy}
          onClick={() => window.dispatchEvent(new CustomEvent('start-playback'))}
        >
          Play
        </button>
      )}

      <button
        className="btn btn-ghost btn-sm w-full"
        onClick={() => {
          clearAll();
          setParam('planning', false);
          setParam('playing', false);
          setParam('canPlay', false);
          setParam('statusText', 'Ready');
        }}
      >
        Clear All
      </button>
    </div>
  );
}
