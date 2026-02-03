import React from 'react';
import type { RunStatus, RunState } from '../../types/dashboard';
import { formatDuration, formatDateTime } from '../../utils/time';
import { Activity, Clock, Database, Layers, Hash } from 'lucide-react';
import clsx from 'clsx';
import './RunHeader.css';

interface RunHeaderProps {
    status: RunStatus;
}

const StateBadge: React.FC<{ state: RunState }> = ({ state }) => {
    const colorClass = clsx({
        'bg-success': state === 'RUNNING' || state === 'COMPLETED',
        'bg-warning': state === 'PAUSED',
        'bg-danger': state === 'HALTED' || state === 'DEGRADED',
    });

    return (
        <span className={`state-badge ${colorClass} text-sm font-bold`}>
            {state}
        </span>
    );
};

export const RunHeader: React.FC<RunHeaderProps> = ({ status }) => {
    return (
        <div className="run-header bg-surface rounded border p-4 mb-4">
            <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-4">
                    <h1 className="text-lg font-bold flex items-center gap-2">
                        <Hash size={20} />
                        {status.runId}
                    </h1>
                    <StateBadge state={status.state} />
                </div>
                <div className="text-sm text-muted">
                    Last updated: {formatDateTime(status.lastUpdated)}
                </div>
            </div>

            <div className="metrics-grid">
                <div className="metric-item">
                    <label className="text-sm text-muted flex items-center gap-2">
                        <Layers size={16} /> Model
                    </label>
                    <div className="font-mono">{status.modelName}</div>
                </div>
                <div className="metric-item">
                    <label className="text-sm text-muted flex items-center gap-2">
                        <Activity size={16} /> Phase
                    </label>
                    <div className="font-mono">{status.phase}</div>
                </div>
                <div className="metric-item">
                    <label className="text-sm text-muted flex items-center gap-2">
                        <Database size={16} /> Tokens
                    </label>
                    <div className="font-mono">{status.tokensProcessed.toLocaleString()}</div>
                </div>
                <div className="metric-item">
                    <label className="text-sm text-muted flex items-center gap-2">
                        Step
                    </label>
                    <div className="font-mono">{status.currentStep.toLocaleString()}</div>
                </div>
                <div className="metric-item">
                    <label className="text-sm text-muted flex items-center gap-2">
                        <Clock size={16} /> Runtime
                    </label>
                    <div className="font-mono">{formatDuration(status.wallClockRuntimeSeconds)}</div>
                </div>
            </div>
        </div>
    );
};
