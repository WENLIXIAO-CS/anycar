import { usePlannerStore } from '../../stores/plannerStore';
import { ParamSlider } from './ParamSlider';

export function DynamicsSelector() {
  const dynamics = usePlannerStore((s) => s.dynamics);
  const dynamicsKwargs = usePlannerStore((s) => s.dynamicsKwargs);
  const dynamicsInfo = usePlannerStore((s) => s.dynamicsInfo);
  const setParam = usePlannerStore((s) => s.setParam);
  const updateDynamicsKwargs = usePlannerStore((s) => s.updateDynamicsKwargs);

  const kwargEntries = Object.entries(dynamicsKwargs);

  return (
    <div className="form-control">
      <label className="label py-0.5">
        <span className="label-text text-xs">Dynamics Model</span>
      </label>
      <select
        className="select select-bordered select-xs w-full"
        value={dynamics}
        onChange={(e) => {
          setParam('dynamics', e.target.value);
          updateDynamicsKwargs();
        }}
      >
        {Object.keys(dynamicsInfo).length > 0
          ? Object.keys(dynamicsInfo).map((key) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))
          : <option value={dynamics}>{dynamics}</option>
        }
      </select>

      {kwargEntries.length > 0 && (
        <div className="collapse collapse-arrow bg-base-300 overflow-hidden mt-2">
          <input type="checkbox" defaultChecked />
          <div className="collapse-title text-sm font-medium">Dynamics Params</div>
          <div className="collapse-content flex flex-col gap-1">
            {kwargEntries.map(([key, val]) => (
              <ParamSlider
                key={key}
                label={key}
                value={val}
                min={0.01}
                max={2.0}
                step={0.01}
                onChange={(v) =>
                  setParam('dynamicsKwargs', { ...dynamicsKwargs, [key]: v })
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
