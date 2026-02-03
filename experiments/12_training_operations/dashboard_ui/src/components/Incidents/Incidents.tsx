import React from 'react';
import type { Incident } from '../../types/dashboard';
import { AlertTriangle, CheckCircle, PauseCircle, ShieldAlert } from 'lucide-react';
import { formatDateTime } from '../../utils/time';
import clsx from 'clsx';

interface IncidentsProps {
    incidents: Incident[];
}

const SeverityBadge: React.FC<{ severity: string }> = ({ severity }) => {
    const badgeClass = clsx({
        'badge-danger': severity === 'SEV-1',
        'badge-warning': severity === 'SEV-2',
        'badge-neutral': severity === 'SEV-3',
    });
    return <span className={`badge ${badgeClass}`}>{severity}</span>;
};

const StatusIcon: React.FC<{ status: string }> = ({ status }) => {
    if (status === 'RESOLVED') return <CheckCircle size={16} className="text-success" />;
    if (status === 'PAUSED') return <PauseCircle size={16} className="text-warning" />;
    return <AlertTriangle size={16} className="text-danger animate-pulse" />;
};

export const Incidents: React.FC<IncidentsProps> = ({ incidents }) => {
    return (
        <div className="glass-panel p-6">
            <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                <ShieldAlert className="text-warning" size={20} />
                Operational Incidents
            </h3>

            {incidents.length === 0 ? (
                <div className="text-sm text-dim italic text-center py-4">
                    No active incidents reported.
                </div>
            ) : (
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="text-xs text-muted uppercase border-b border-border/50">
                                <th className="pb-2 pl-2">Severity</th>
                                <th className="pb-2">Event Type</th>
                                <th className="pb-2">Source</th>
                                <th className="pb-2">Status</th>
                                <th className="pb-2 text-right pr-2">Logged</th>
                            </tr>
                        </thead>
                        <tbody className="text-sm">
                            {incidents.map((inc) => (
                                <tr key={inc.id} className="border-b border-border/10 last:border-0 hover:bg-white/5 transition-colors">
                                    <td className="py-3 pl-2"><SeverityBadge severity={inc.severity} /></td>
                                    <td className="py-3 font-medium text-white">{inc.eventType}</td>
                                    <td className="py-3 text-dim">{inc.source}</td>
                                    <td className="py-3">
                                        <div className="flex items-center gap-2">
                                            <StatusIcon status={inc.status} />
                                            <span className={clsx("text-xs font-bold", inc.status === 'RESOLVED' ? "text-success" : "text-white")}>
                                                {inc.status}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="py-3 text-right pr-2 text-muted font-mono text-xs">
                                        {formatDateTime(inc.timestamp)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};
