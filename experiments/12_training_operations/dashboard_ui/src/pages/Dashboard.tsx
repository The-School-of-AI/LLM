import React, { useState, useEffect } from 'react';
import { RunHeader } from '../components/RunHeader/RunHeader';
import { LiveSignals } from '../components/LiveSignals/LiveSignals';
import { Checkpoints } from '../components/Checkpoints/Checkpoints';
import { Incidents } from '../components/Incidents/Incidents';
import { CostDrift } from '../components/CostDrift/CostDrift';
import type { RunStatus, LiveMetrics, CheckpointStatus, Incident, CostStatus } from '../types/dashboard';
import { AlertTriangle, RefreshCw, Loader } from 'lucide-react';

const Dashboard: React.FC = () => {
    const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
    const [liveMetrics, setLiveMetrics] = useState<LiveMetrics | null>(null);
    const [checkpointStatus, setCheckpointStatus] = useState<CheckpointStatus | null>(null);
    const [incidents, setIncidents] = useState<Incident[]>([]);
    const [costStatus, setCostStatus] = useState<CostStatus | null>(null);

    const [lastFetchTime, setLastFetchTime] = useState<Date | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(true);

    const isStale = lastFetchTime ? (Date.now() - lastFetchTime.getTime() > 60000) : false;

    const fetchData = async () => {
        setLoading(true);
        try {
            setError(null);
            console.log('[Dashboard] Fetching data...');
            const responses = await Promise.all([
                fetch('/api/run_status.json'),
                fetch('/api/live_metrics.json'),
                fetch('/api/checkpoints.json'),
                fetch('/api/incidents.json'),
                fetch('/api/cost.json')
            ]);

            // Helper to safely parse JSON
            const parseJson = async (res: Response) => {
                if (!res.ok) throw new Error(`HTTP error ${res.status}`);
                const text = await res.text();
                try {
                    return JSON.parse(text);
                } catch (e) {
                    console.error('[Dashboard] JSON Parse Error:', text.substring(0, 50));
                    throw new Error('Invalid JSON response (likely HTML fallback)');
                }
            };

            const [runStatusData, liveMetricsData, checkpointData, incidentsData, costData] = await Promise.all(
                responses.map(parseJson)
            );

            setRunStatus(runStatusData);
            setLiveMetrics(liveMetricsData);
            setCheckpointStatus(checkpointData);
            // Handle different response structures for incidents
            setIncidents(Array.isArray(incidentsData) ? incidentsData : (incidentsData.incidents || []));
            setCostStatus(costData);

            setLastFetchTime(new Date());

        } catch (e) {
            console.error('[Dashboard] Data fetch error:', e);
            setError("Could not connect to data endpoints. Ensure mock server is running.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 30000); // 30s polling
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="min-h-screen bg-bg text-text p-6">

            {/* Run Header with Loading/Error integration */}
            <div className="mb-6 flex justify-between items-start">
                {runStatus ? (
                    <div className="w-full"><RunHeader status={runStatus} /></div>
                ) : (
                    <div className="p-4 border rounded w-full flex items-center justify-between">
                        <span className="text-muted font-bold">Training Dashboard (v0)</span>
                        {loading && <div className="flex items-center gap-2 text-sm text-muted"><Loader className="animate-spin" size={16} /> Connecting...</div>}
                    </div>
                )}
            </div>

            {/* Error / Stale Banners */}
            <div className="mb-4 space-y-2">
                {error && (
                    <div className="bg-danger/10 text-danger border border-danger p-2 rounded flex items-center gap-2">
                        <AlertTriangle size={16} />
                        <strong>Connection Error:</strong> {error}
                        <button onClick={fetchData} className="ml-auto flex items-center gap-1 text-sm underline">
                            <RefreshCw size={12} /> Retry
                        </button>
                    </div>
                )}
                {isStale && !error && (
                    <div className="bg-warning/20 text-warning border border-warning p-2 rounded flex items-center gap-2">
                        <AlertTriangle size={16} />
                        <strong>Data Stale:</strong> No updates received for over 60 seconds.
                        <button onClick={fetchData} className="ml-auto flex items-center gap-1 text-sm underline">
                            <RefreshCw size={12} /> Retry
                        </button>
                    </div>
                )}
            </div>

            {/* Layout Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Column: Live Signals (Width 2/3 roughly or full width on small) */}
                <div className="lg:col-span-2">
                    {liveMetrics ? <LiveSignals metrics={liveMetrics} /> : (
                        <div className="p-8 border rounded text-center text-muted h-64 flex items-center justify-center">
                            {loading ? "Loading Live Signals..." : "No Live Signals"}
                        </div>
                    )}
                </div>

                {/* Right Column: Checkpoints, Incidents, Cost */}
                <div className="flex flex-col gap-6">
                    <Checkpoints status={checkpointStatus} />
                    <Incidents incidents={incidents} />
                    {costStatus ? <CostDrift status={costStatus} /> : <div className="p-4 border rounded text-muted">No Cost Data</div>}
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
