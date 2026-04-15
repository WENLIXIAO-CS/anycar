import { PresetsSection } from './PresetsSection';
import { DynamicsSelector } from './DynamicsSelector';
import { MppiParamsSection } from './MppiParamsSection';
import { CostWeightsSection } from './CostWeightsSection';
import { MapSettingsSection } from './MapSettingsSection';
import { DrawingTools } from './DrawingTools';

export function LeftPanel() {
  return (
    <aside className="bg-base-200 overflow-y-auto overscroll-contain border-r border-base-300">
      <div className="p-4 flex flex-col gap-3">
        <PresetsSection />
        <DynamicsSelector />
        <MppiParamsSection />
        <CostWeightsSection />
        <MapSettingsSection />
        <DrawingTools />
      </div>
    </aside>
  );
}
