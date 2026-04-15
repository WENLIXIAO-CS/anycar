import { useCanvasStore } from '../../stores/canvasStore';
import type { CanvasState } from '../../stores/canvasStore';

const TOOLS: { mode: CanvasState['drawMode']; icon: string; activeClass: string }[] = [
  { mode: 'rect', icon: '▭', activeClass: 'btn-primary' },
  { mode: 'circle', icon: '○', activeClass: 'btn-primary' },
  { mode: 'start', icon: '▶', activeClass: 'btn-success' },
  { mode: 'goal', icon: '★', activeClass: 'btn-error' },
  { mode: 'erase', icon: '✕', activeClass: 'btn-warning' },
];

export function DrawingTools() {
  const drawMode = useCanvasStore((s) => s.drawMode);
  const setDrawMode = useCanvasStore((s) => s.setDrawMode);

  return (
    <div className="form-control">
      <label className="label py-0.5">
        <span className="label-text text-xs">Drawing Tools</span>
      </label>
      <div className="join w-full">
        {TOOLS.map(({ mode, icon, activeClass }) => (
          <button
            key={mode}
            className={`btn btn-sm join-item flex-1 ${
              drawMode === mode ? activeClass : 'btn-ghost'
            }`}
            onClick={() => setDrawMode(mode)}
            title={mode}
          >
            {icon}
          </button>
        ))}
      </div>
    </div>
  );
}
