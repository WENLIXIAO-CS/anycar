import { PRESETS } from '../../constants/presets';
import { usePlannerStore } from '../../stores/plannerStore';

const PRESET_BUTTONS = [
  { key: 'straight', label: 'Straight' },
  { key: 'lane_change', label: 'Lane Change' },
  { key: 's_curve', label: 'S-Curve' },
  { key: 'slalom', label: 'Slalom' },
] as const;

export function PresetsSection() {
  return (
    <div className="collapse collapse-arrow bg-base-300 overflow-hidden">
      <input type="checkbox" defaultChecked />
      <div className="collapse-title text-sm font-medium min-h-0 py-2">Presets</div>
      <div className="collapse-content !pb-3">
        <div className="grid grid-cols-2 gap-2">
          {PRESET_BUTTONS.map(({ key, label }) => (
            <button
              key={key}
              className="btn btn-sm btn-outline"
              onClick={() => {
                usePlannerStore.getState().loadPreset(key);
                const preset = PRESETS[key];
                if (preset) {
                  window.dispatchEvent(
                    new CustomEvent('load-preset-scene', { detail: preset.scene })
                  );
                }
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
