import StatusBar from './StatusBar';
import DistanceChart from './DistanceChart';
import ActionChart from './ActionChart';

export default function RightPanel() {
  return (
    <aside className="bg-base-200 p-3 flex flex-col gap-2 border-l border-base-300 overflow-y-auto overscroll-contain">
      <StatusBar />
      <div className="bg-base-300 rounded-lg p-2">
        <h3 className="text-xs font-semibold opacity-60 mb-1">Distance to Goal</h3>
        <div style={{ height: 140, position: 'relative' }}>
          <DistanceChart />
        </div>
      </div>
      <div className="bg-base-300 rounded-lg p-2">
        <h3 className="text-xs font-semibold opacity-60 mb-1">Actions</h3>
        <div style={{ height: 140, position: 'relative' }}>
          <ActionChart />
        </div>
      </div>
    </aside>
  );
}
