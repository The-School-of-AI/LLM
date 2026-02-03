import React from 'react';
import type { CheckpointStatus } from '../../types/dashboard';
import { formatDateTime, getTimeAgo } from '../../utils/time';
import { Save, AlertTriangle, Lock, Unlock } from 'lucide-react';
import clsx from 'clsx';
import './Checkpoints.css';

interface CheckpointsProps {
    status: CheckpointStatus | null;
}

export const Checkpoints: React.FC<CheckpointsProps> = ({ status }) => {
    if (!status) {
        return (
            <div className="bg-surface border rounded p-4 mb-4 border-warning text-warning flex items-center gap-4">
                <AlertTriangle size={24} />
                <div>
                    <h3 className="font-bold">Checkpoint Status Unknown</h3>
                    <p className="text-sm">No checkpoint data available.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-surface border rounded p-4 mb-4">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold flex items-center gap-2">
                    <Save size={20} /> Checkpoints
                </h2>
                <div className="text-sm text-muted">
                    Last updated: {formatDateTime(status.lastUpdated)}
                </div>
            </div>

            <div className="checkpoints-grid">
                <div className="checkpoint-item">
                    <label className="text-sm text-muted">Last Checkpoint Step</label>
                    <div className="font-mono text-lg">{status.lastCheckpointStep.toLocaleString()}</div>
                </div>

                <div className="checkpoint-item">
                    <label className="text-sm text-muted">Time</label>
                    <div className="text-sm">{formatDateTime(status.checkpointTimestamp)}</div>
                    <div className="text-xs text-muted">({getTimeAgo(status.checkpointTimestamp)})</div>
                </div>

                <div className="checkpoint-item">
                    <label className="text-sm text-muted">Checkpoint ID</label>
                    <div className="font-mono text-sm truncate" title={status.checkpointId}>
                        {status.checkpointId.substring(0, 8)}...
                    </div>
                </div>

                <div className="checkpoint-item">
                    <label className="text-sm text-muted">Optimizer State</label>
                    <div className={clsx("font-bold", status.hasOptimizerState ? "text-success" : "text-warning")}>
                        {status.hasOptimizerState ? "SAVED" : "NOT SAVED"}
                    </div>
                </div>

                <div className="checkpoint-item">
                    <label className="text-sm text-muted">Growth Phase</label>
                    <div>{status.growthPhase || '-'}</div>
                </div>

                <div className="checkpoint-item">
                    <label className="text-sm text-muted">LoRA Lock</label>
                    <div className="flex items-center gap-2">
                        {status.loraLockSummary ? (
                            <>
                                <Lock size={14} className="text-warning" />
                                <span>{status.loraLockSummary}</span>
                            </>
                        ) : (
                            <>
                                <Unlock size={14} className="text-muted" />
                                <span className="text-muted">Unlocked</span>
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
