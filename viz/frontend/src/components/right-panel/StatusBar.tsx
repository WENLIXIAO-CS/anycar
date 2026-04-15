import { usePlannerStore } from '../../stores/plannerStore';

export default function StatusBar() {
  const currentStep = usePlannerStore((s) => s.currentStep);
  const posError = usePlannerStore((s) => s.posError);
  const statusText = usePlannerStore((s) => s.statusText);

  return (
    <div className="flex gap-2 text-xs">
      <div className="badge badge-ghost flex-1 justify-between px-2 py-3">
        <span className="opacity-60">Step</span>
        <span className="font-mono">{currentStep}</span>
      </div>
      <div className="badge badge-ghost flex-1 justify-between px-2 py-3">
        <span className="opacity-60">Err</span>
        <span className="font-mono">{posError !== null ? posError.toFixed(3) : '—'}</span>
      </div>
      <div className="badge badge-ghost flex-1 justify-between px-2 py-3">
        <span className="opacity-60">{statusText}</span>
      </div>
    </div>
  );
}
