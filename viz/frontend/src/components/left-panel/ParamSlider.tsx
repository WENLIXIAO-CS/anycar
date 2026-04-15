interface ParamSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  colorClass?: string;
}

export function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  colorClass = 'range-primary',
}: ParamSliderProps) {
  return (
    <div className="form-control">
      <label className="label py-0.5">
        <span className="label-text text-xs">{label}</span>
        <span className="label-text-alt text-xs font-mono">{value}</span>
      </label>
      <input
        type="range"
        className={`range range-xs ${colorClass}`}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
