import { usePlannerStore } from '../../stores/plannerStore';
import { ParamSlider } from './ParamSlider';

export function MppiParamsSection() {
  const backend = usePlannerStore((s) => s.backend);
  const dt = usePlannerStore((s) => s.dt);
  const nRollouts = usePlannerStore((s) => s.nRollouts);
  const sigma = usePlannerStore((s) => s.sigma);
  const steerSigma = usePlannerStore((s) => s.steerSigma);
  const lam = usePlannerStore((s) => s.lam);
  const horizonKnots = usePlannerStore((s) => s.horizonKnots);
  const numIntermediate = usePlannerStore((s) => s.numIntermediate);
  const setParam = usePlannerStore((s) => s.setParam);

  return (
    <div className="collapse collapse-arrow bg-base-300 overflow-hidden">
      <input type="checkbox" defaultChecked />
      <div className="collapse-title text-sm font-medium">MPPI Parameters</div>
      <div className="collapse-content flex flex-col gap-1">
        <div className="form-control">
          <label className="label py-0.5">
            <span className="label-text text-xs">Backend</span>
            <span className="label-text-alt text-xs font-mono">{backend}</span>
          </label>
          <select
            className="select select-bordered select-xs w-full"
            value={backend}
            onChange={(e) => setParam('backend', e.target.value)}
          >
            <option value="spline">spline</option>
            <option value="torch">torch</option>
          </select>
        </div>

        <ParamSlider label="dt" value={dt} min={0.01} max={0.5} step={0.01} onChange={(v) => setParam('dt', v)} colorClass="range-primary" />
        <ParamSlider label="n_rollouts" value={nRollouts} min={256} max={4096} step={256} onChange={(v) => setParam('nRollouts', v)} colorClass="range-primary" />
        <ParamSlider label="sigma" value={sigma} min={0.01} max={1.0} step={0.01} onChange={(v) => setParam('sigma', v)} colorClass="range-primary" />
        <ParamSlider label="steer_sigma" value={steerSigma} min={0.005} max={0.3} step={0.005} onChange={(v) => setParam('steerSigma', v)} colorClass="range-primary" />
        <ParamSlider label="lambda" value={lam} min={0.001} max={0.5} step={0.001} onChange={(v) => setParam('lam', v)} colorClass="range-primary" />
        <ParamSlider label="horizon_knots" value={horizonKnots} min={2} max={20} step={1} onChange={(v) => setParam('horizonKnots', v)} colorClass="range-primary" />
        <ParamSlider label="num_intermediate" value={numIntermediate} min={1} max={10} step={1} onChange={(v) => setParam('numIntermediate', v)} colorClass="range-primary" />
      </div>
    </div>
  );
}
