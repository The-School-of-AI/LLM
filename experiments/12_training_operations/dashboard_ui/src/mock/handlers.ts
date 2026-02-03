import { http, HttpResponse } from 'msw';
import type { RunStatus, LiveMetrics, CheckpointStatus, Incident, CostStatus } from '../types/dashboard';

// --- Simulation State ---
let currentStep = 12500;
const startTime = Date.now() - (4 * 60 * 60 * 1000); // Started 4 hours ago

// Helper to generate time series data
function generateTimeSeries(points: number, baseValue: number, variance: number): { timestamp: string; value: number }[] {
    const now = Date.now();
    const data = [];
    for (let i = points; i >= 0; i--) {
        const time = now - (i * 30 * 1000); // 30s intervals
        data.push({
            timestamp: new Date(time).toISOString(),
            value: baseValue + (Math.random() * variance * 2 - variance),
        });
    }
    return data;
}

export const handlers = [
    // 1. Run Status
    http.get('/api/run_status.json', () => {
        currentStep += Math.floor(Math.random() * 5); // Simulate progress
        const now = new Date();

        const status: RunStatus = {
            runId: 'run-v4-7b-ctx8k',
            modelName: 'Llama-3-70b-v4',
            phase: 'post-train',
            state: 'RUNNING',
            currentStep: currentStep,
            tokensProcessed: currentStep * 4096 * 16, // Assuming 4k context * 16 BS
            wallClockRuntimeSeconds: (now.getTime() - startTime) / 1000,
            lastUpdated: now.toISOString(),
        };
        return HttpResponse.json(status);
    }),

    // 2. Live Metrics
    http.get('/api/live_metrics.json', () => {
        const now = new Date();
        const metrics: LiveMetrics = {
            loss: generateTimeSeries(20, 1.4, 0.05),
            throughput: generateTimeSeries(20, 45000, 2000),
            gpuUtilization: generateTimeSeries(20, 92, 5),
            gpuMemory: generateTimeSeries(20, 78, 2),
            routingHealth: 'OK',
            lastUpdated: now.toISOString(),
        };
        return HttpResponse.json(metrics);
    }),

    // 3. Checkpoints
    http.get('/api/checkpoints.json', () => {
        const now = new Date();
        const lastCheckpointTime = new Date(now.getTime() - 15 * 60 * 1000); // 15 mins ago

        const checkpoint: CheckpointStatus = {
            lastCheckpointStep: Math.floor(currentStep / 1000) * 1000,
            checkpointTimestamp: lastCheckpointTime.toISOString(),
            checkpointId: `ckpt-${Math.floor(currentStep / 1000) * 1000}`,
            hasOptimizerState: true,
            growthPhase: null,
            loraLockSummary: 'Locked (Step 10000)',
            lastUpdated: now.toISOString(),
        };
        return HttpResponse.json(checkpoint);
    }),

    // 4. Incidents
    http.get('/api/incidents.json', () => {
        const incidents: Incident[] = [
            {
                id: 'inc-001',
                severity: 'SEV-3',
                eventType: 'Throughput degradation',
                timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
                status: 'RESOLVED',
                escalationTarget: 'On-call (M. Smith)'
            }
        ];
        return HttpResponse.json({ incidents }); // Note: Dashboard expects array directly or inside object? Let's check logic.
        // Dashboard.tsx: setIncidents(data.incidents || data); -> Handles both.
    }),

    // 5. Cost
    http.get('/api/cost.json', () => {
        const now = new Date();
        const cost: CostStatus = {
            currentBurnRate: 450.50,
            expectedSpend: 15000,
            actualSpend: 15200,
            driftStatus: 'OK',
            haltProximity: 15,
            lastUpdated: now.toISOString(),
        };
        return HttpResponse.json(cost);
    }),
];
