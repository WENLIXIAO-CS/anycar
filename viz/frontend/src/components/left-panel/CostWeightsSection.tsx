import { usePlannerStore } from '../../stores/plannerStore';
import { ParamSlider } from './ParamSlider';

export function CostWeightsSection() {
  const goalPosWeight = usePlannerStore((s) => s.goalPosWeight);
  const goalHeadingWeight = usePlannerStore((s) => s.goalHeadingWeight);
  const goalVelWeight = usePlannerStore((s) => s.goalVelWeight);
  const collisionWeight = usePlannerStore((s) => s.collisionWeight);
  const collisionDecay = usePlannerStore((s) => s.collisionDecay);
  const collisionInsideWeight = usePlannerStore((s) => s.collisionInsideWeight);
  const smoothnessWeight = usePlannerStore((s) => s.smoothnessWeight);
  const terminalWeight = usePlannerStore((s) => s.terminalWeight);
  const controlRegWeight = usePlannerStore((s) => s.controlRegWeight);
  const signChangeWeight = usePlannerStore((s) => s.signChangeWeight);
  const gammaSigma = usePlannerStore((s) => s.gammaSigma);
  const covFloor = usePlannerStore((s) => s.covFloor);
  const setParam = usePlannerStore((s) => s.setParam);

  const color = 'range-secondary';

  return (
    <div className="collapse collapse-arrow bg-base-300 overflow-hidden">
      <input type="checkbox" defaultChecked />
      <div className="collapse-title text-sm font-medium">Cost Weights</div>
      <div className="collapse-content flex flex-col gap-1">
        <ParamSlider label="goal_pos_weight" value={goalPosWeight} min={0} max={10000} step={50} onChange={(v) => setParam('goalPosWeight', v)} colorClass={color} />
        <ParamSlider label="goal_heading_weight" value={goalHeadingWeight} min={0} max={500} step={1} onChange={(v) => setParam('goalHeadingWeight', v)} colorClass={color} />
        <ParamSlider label="goal_vel_weight" value={goalVelWeight} min={0} max={500} step={5} onChange={(v) => setParam('goalVelWeight', v)} colorClass={color} />
        <ParamSlider label="collision_weight" value={collisionWeight} min={0} max={500} step={1} onChange={(v) => setParam('collisionWeight', v)} colorClass={color} />
        <ParamSlider label="collision_decay" value={collisionDecay} min={0.1} max={20} step={0.1} onChange={(v) => setParam('collisionDecay', v)} colorClass={color} />
        <ParamSlider label="collision_inside_weight" value={collisionInsideWeight} min={0} max={2000} step={10} onChange={(v) => setParam('collisionInsideWeight', v)} colorClass={color} />
        <ParamSlider label="smoothness_weight" value={smoothnessWeight} min={0} max={100} step={1} onChange={(v) => setParam('smoothnessWeight', v)} colorClass={color} />
        <ParamSlider label="terminal_weight" value={terminalWeight} min={0} max={50} step={1} onChange={(v) => setParam('terminalWeight', v)} colorClass={color} />
        <ParamSlider label="control_reg_weight" value={controlRegWeight} min={0} max={5} step={0.05} onChange={(v) => setParam('controlRegWeight', v)} colorClass={color} />
        <ParamSlider label="sign_change_weight" value={signChangeWeight} min={0} max={500} step={5} onChange={(v) => setParam('signChangeWeight', v)} colorClass={color} />
        <ParamSlider label="gamma_sigma" value={gammaSigma} min={0} max={1.0} step={0.05} onChange={(v) => setParam('gammaSigma', v)} colorClass={color} />
        <ParamSlider label="cov_floor" value={covFloor} min={0} max={0.05} step={0.001} onChange={(v) => setParam('covFloor', v)} colorClass={color} />
      </div>
    </div>
  );
}
