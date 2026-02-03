import React from 'react';
import type { CostStatus } from '../../types/dashboard';
import { formatDateTime } from '../../utils/time';
import { DollarSign, TrendingUp, AlertTriangle } from 'lucide-react';
import clsx from 'clsx';
import './CostDrift.css';

interface CostDriftProps {
    status: CostStatus;
}

export const CostDrift: React.FC<CostDriftProps> = ({ status }) => {
    const isDriftWarning = status.driftStatus === 'WARNING';

    return (
        <div className="bg-surface border rounded p-4">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold flex items-center gap-2">
                    <DollarSign size={20} /> Cost & Drift
                </h2>
                <div className="text-sm text-muted">
                    Last updated: {formatDateTime(status.lastUpdated)}
                </div>
            </div>

            <div className="cost-grid">
                <div className="cost-item">
                    <label className="text-sm text-muted">Current Burn Rate</label>
                    <div className="font-mono text-xl">${status.currentBurnRate.toLocaleString()}/hr</div>
                </div>

                <div className="cost-item">
                    <label className="text-sm text-muted">Expected vs Actual</label>
                    <div className="flex flex-col gap-1">
                        <div className="flex justify-between text-sm">
                            <span className="text-muted">Exp:</span>
                            <span className="font-mono">${status.expectedSpend.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                            <span className="text-muted">Act:</span>
                            <span className="font-mono">${status.actualSpend.toLocaleString()}</span>
                        </div>
                    </div>
                </div>

                <div className="cost-item">
                    <label className="text-sm text-muted">Drift Status</label>
                    <div className={clsx("font-bold flex items-center gap-2", isDriftWarning ? "text-warning" : "text-success")}>
                        {isDriftWarning ? <AlertTriangle size={16} /> : <TrendingUp size={16} />}
                        {status.driftStatus}
                    </div>
                </div>

                {status.haltProximity !== undefined && (
                    <div className="cost-item">
                        <label className="text-sm text-muted">Halt Proximity</label>
                        <div className="w-full bg-slate-700 rounded-full h-2.5 mt-2">
                            <div
                                className={clsx("h-2.5 rounded-full", status.haltProximity > 80 ? "bg-danger" : "bg-primary")}
                                style={{ width: `${Math.min(status.haltProximity, 100)}%` }}
                            ></div>
                        </div>
                        <div className="text-right text-xs text-muted mt-1">{status.haltProximity}%</div>
                    </div>
                )}
            </div>
        </div>
    );
};
