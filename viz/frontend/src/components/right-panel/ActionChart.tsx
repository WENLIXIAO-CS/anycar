import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useChartStore } from '../../stores/chartStore';
import { ACTION_COLORS } from '../../constants/dynamics';

export default function ActionChart() {
  const actionData = useChartStore((s) => s.actionData);
  const actionLabels = useChartStore((s) => s.actionLabels);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={actionData} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.05)" />
        <XAxis
          dataKey="step"
          tick={{ fill: '#aaa', fontSize: 10 }}
          tickLine={{ stroke: '#aaa' }}
          axisLine={{ stroke: '#aaa' }}
        />
        <YAxis
          tick={{ fill: '#aaa', fontSize: 10 }}
          tickLine={{ stroke: '#aaa' }}
          axisLine={{ stroke: '#aaa' }}
        />
        <Legend
          wrapperStyle={{ fontSize: 10 }}
          iconSize={8}
        />
        {actionLabels.map((label, i) => (
          <Line
            key={label}
            type="monotone"
            dataKey={label}
            stroke={ACTION_COLORS[i % ACTION_COLORS.length]}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
