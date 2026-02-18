import React from 'react';
import type { RunStatus, RunState } from '../../types/dashboard';
import { formatDuration, formatDateTime } from '../../utils/time';
import { Clock, Database } from 'lucide-react';
import clsx from 'clsx';
import './RunHeader.css';

interface RunHeaderProps {
    status: RunStatus;
}

const StateBadge: React.FC<{ state: RunState }> = ({ state }) => {
    const badgeClass = clsx({
        'badge-success': state === 'RUNNING' || state === 'COMPLETED',
        'badge-warning': state === 'PAUSED',
        'badge-danger': state === 'HALTED' || state === 'DEGRADED',
    });

    return (
        <span className={`badge ${badgeClass} ml-3`}>
            {state}
        </span>
    );
};

export const RunHeader: React.FC<RunHeaderProps> = ({ status }) => {
    return (
        <div className="glass-panel p-6 mb-6">
            <div className="flex justify-between items-start mb-6">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-bold text-primary uppercase tracking-wider">Active Run</span>
                        <div className="h-px bg-border flex-1 w-12"></div>
                    </div>
                    <h1 className="text-2xl font-bold flex items-center">
                        <span className="text-gradient mr-3">{status.modelName}</span>
                        <span className="text-muted text-lg font-normal opacity-70">/ {status.runId}</span>
                        <StateBadge state={status.state} />
                    </h1>
                    <div className="flex items-center gap-4 mt-2 text-sm text-muted">
                        <span className="flex items-center gap-1.5">
                            <Database size={14} className="text-accent" />
                            {status.source}
                        </span>
                        <span className="flex items-center gap-1.5">
                            <Clock size={14} />
                            Last updated: {formatDateTime(status.lastUpdated)}
                        </span>
                    </div>
                </div>

                <div className="text-right">
                    <div className="text-xs text-muted uppercase tracking-wider mb-1">Runtime</div>
                    <div className="font-mono text-xl font-bold">{formatDuration(status.wallClockRuntimeSeconds)}</div>
                </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-white/5">
                <div className="flex flex-col">
                    <span className="text-xs text-muted uppercase mb-1">Phase</span>
                    <span className="font-mono font-bold text-lg">{status.phase}</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-xs text-muted uppercase mb-1">Step</span>
                    <span className="font-mono font-bold text-lg">{status.currentStep.toLocaleString()}</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-xs text-muted uppercase mb-1">Tokens</span>
                    <span className="font-mono font-bold text-lg">{status.tokensProcessed.toLocaleString()}</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-xs text-muted uppercase mb-1">Est. Completion</span>
                    <span className="font-mono font-bold text-lg text-dim">--</span>
                </div>
            </div>
        </div>
    );
};
