import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
} from 'recharts';
import { useChartStore } from '../../stores/chartStore';

export default function DistanceChart() {
  const distanceData = useChartStore((s) => s.distanceData);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={distanceData} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
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
        <Line
          type="monotone"
          dataKey="posError"
          stroke="#38bdf8"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="tolerance"
          stroke="#22c55e"
          strokeWidth={1}
          strokeDasharray="4 3"
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
