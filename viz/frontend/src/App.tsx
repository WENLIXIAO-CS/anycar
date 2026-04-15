import { useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { MainLayout } from './components/MainLayout';
import { FloatingToolbar } from './components/FloatingToolbar';
import { usePlannerStore } from './stores/plannerStore';

export default function App() {
  const fetchDynamicsInfo = usePlannerStore(s => s.fetchDynamicsInfo);

  useEffect(() => {
    fetchDynamicsInfo();
  }, [fetchDynamicsInfo]);

  return (
    <div className="h-screen bg-base-100 flex flex-col overflow-hidden">
      <Navbar />
      <MainLayout />
      <FloatingToolbar />
    </div>
  );
}
