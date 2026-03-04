import { useSSE } from './hooks/useSSE'
import { useMetricsStore } from './stores/metricsStore'
import { Topbar } from './components/Topbar'
import { LossMetrics } from './components/LossMetrics'
import { ThroughputSection } from './components/ThroughputSection'
import { ArchitectureStats } from './components/ArchitectureStats'
import { MoEAnalytics } from './components/MoEAnalytics'
import { TimelineSection } from './components/TimelineSection'
import { InfrastructureSection } from './components/InfrastructureSection'
import { CheckpointsSection } from './components/CheckpointsSection'
import { GeneratedSamples } from './components/GeneratedSamples'

export default function App() {
  useSSE()
  const selectedTab = useMetricsStore(s => s.selectedTab)

  return (
    <>
      <Topbar />
      <div className="dash">
        {selectedTab === 'overview' && (
          <>
            <LossMetrics />
            <ThroughputSection />
          </>
        )}
        {selectedTab === 'architecture' && (
          <>
            <ArchitectureStats />
            <MoEAnalytics />
          </>
        )}
        {selectedTab === 'milestones' && (
          <>
            <TimelineSection />
            <CheckpointsSection />
            <GeneratedSamples />
          </>
        )}
        {selectedTab === 'infrastructure' && (
          <InfrastructureSection />
        )}
      </div>
    </>
  )
}
