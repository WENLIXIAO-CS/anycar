import { usePlannerStore } from '../../stores/plannerStore';
import { ParamSlider } from './ParamSlider';

export function MapSettingsSection() {
  const useGlobalPlanner = usePlannerStore((s) => s.useGlobalPlanner);
  const globalPlannerType = usePlannerStore((s) => s.globalPlannerType);
  const resolution = usePlannerStore((s) => s.resolution);
  const inflateRadius = usePlannerStore((s) => s.inflateRadius);
  const stopThreshold = usePlannerStore((s) => s.stopThreshold);
  const goalTolerance = usePlannerStore((s) => s.goalTolerance);
  const replanInterval = usePlannerStore((s) => s.replanInterval);
  const maxSteps = usePlannerStore((s) => s.maxSteps);
  const setParam = usePlannerStore((s) => s.setParam);

  const color = 'range-accent';

  return (
    <div className="collapse collapse-arrow bg-base-300 overflow-hidden">
      <input type="checkbox" defaultChecked />
      <div className="collapse-title text-sm font-medium">Map Settings</div>
      <div className="collapse-content flex flex-col gap-1">
        <div className="form-control">
          <label className="label cursor-pointer py-0.5">
            <span className="label-text text-xs">Global Planner</span>
            <input
              type="checkbox"
              className="toggle toggle-xs toggle-accent"
              checked={useGlobalPlanner}
              onChange={(e) => setParam('useGlobalPlanner', e.target.checked)}
            />
          </label>
        </div>

        {useGlobalPlanner && (
          <div className="form-control">
            <label className="label py-0.5">
              <span className="label-text text-xs">Planner Type</span>
              <span className="label-text-alt text-xs font-mono">{globalPlannerType}</span>
            </label>
            <select
              className="select select-bordered select-xs w-full"
              value={globalPlannerType}
              onChange={(e) => setParam('globalPlannerType', e.target.value)}
            >
              <option value="rrt">rrt</option>
              <option value="hybrid_astar">hybrid_astar</option>
            </select>
          </div>
        )}

        <ParamSlider label="resolution" value={resolution} min={0.02} max={0.2} step={0.01} onChange={(v) => setParam('resolution', v)} colorClass={color} />
        <ParamSlider label="inflate_radius" value={inflateRadius} min={0} max={0.5} step={0.01} onChange={(v) => setParam('inflateRadius', v)} colorClass={color} />
        <ParamSlider label="stop_threshold" value={stopThreshold} min={0.01} max={0.5} step={0.01} onChange={(v) => setParam('stopThreshold', v)} colorClass={color} />
        <ParamSlider label="goal_tolerance" value={goalTolerance} min={0.01} max={0.5} step={0.01} onChange={(v) => setParam('goalTolerance', v)} colorClass={color} />
        <ParamSlider label="replan_interval" value={replanInterval} min={1} max={20} step={1} onChange={(v) => setParam('replanInterval', v)} colorClass={color} />
        <ParamSlider label="max_steps" value={maxSteps} min={50} max={1000} step={50} onChange={(v) => setParam('maxSteps', v)} colorClass={color} />
      </div>
    </div>
  );
}
