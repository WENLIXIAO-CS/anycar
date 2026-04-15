import { LeftPanel } from './left-panel/LeftPanel';
import { CenterPanel } from './canvas/CenterPanel';
import RightPanel from './right-panel/RightPanel';

export function MainLayout() {
  return (
    <main
      className="flex-1 min-h-0 grid"
      style={{ gridTemplateColumns: '18rem 1fr 16rem', gridTemplateRows: 'minmax(0, 1fr)' }}
    >
      <LeftPanel />
      <CenterPanel />
      <RightPanel />
    </main>
  );
}
