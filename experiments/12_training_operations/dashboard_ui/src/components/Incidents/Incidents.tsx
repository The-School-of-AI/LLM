import React from 'react';
import type { Incident, IncidentSeverity, IncidentStatus } from '../../types/dashboard';
import { formatDateTime } from '../../utils/time';
import { AlertOctagon, CheckCircle, Clock, AlertCircle } from 'lucide-react';
import clsx from 'clsx';
import './Incidents.css';

interface IncidentsProps {
    incidents: Incident[];
}

const SeverityBadge: React.FC<{ severity: IncidentSeverity }> = ({ severity }) => {
    const colorClass = clsx({
        'bg-danger': severity === 'SEV-1',
        'bg-warning': severity === 'SEV-2',
        'bg-primary': severity === 'SEV-3', // using primary (blue) for low severity
    });
    return (
        <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold text-white ${colorClass}`}>
            {severity}
        </span>
    );
};

const StatusIcon: React.FC<{ status: IncidentStatus }> = ({ status }) => {
    switch (status) {
        case 'OPEN': return <AlertCircle size={16} className="text-danger" />;
        case 'PAUSED': return <Clock size={16} className="text-warning" />;
        case 'RESOLVED': return <CheckCircle size={16} className="text-success" />;
        default: return <Clock size={16} className="text-muted" />;
    }
}

export const Incidents: React.FC<IncidentsProps> = ({ incidents }) => {
    return (
        <div className="bg-surface border rounded p-4 mb-4">
            <h2 className="text-lg font-bold flex items-center gap-2 mb-4">
                <AlertOctagon size={20} /> Incidents & Events
            </h2>

            {incidents.length === 0 ? (
                <div className="text-muted text-center py-4">No recent incidents.</div>
            ) : (
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="text-muted border-b border-border">
                            <tr>
                                <th className="py-2 px-2">Severity</th>
                                <th className="py-2 px-2">Status</th>
                                <th className="py-2 px-2">Timestamp</th>
                                <th className="py-2 px-2">Type</th>
                                <th className="py-2 px-2">Escalation</th>
                            </tr>
                        </thead>
                        <tbody>
                            {incidents.map((inc) => (
                                <tr key={inc.id} className="border-b border-border last:border-0 hover:bg-slate-700/50">
                                    <td className="py-2 px-2"><SeverityBadge severity={inc.severity} /></td>
                                    <td className="py-2 px-2 flex items-center gap-2">
                                        <StatusIcon status={inc.status} /> {inc.status}
                                    </td>
                                    <td className="py-2 px-2 font-mono">{formatDateTime(inc.timestamp)}</td>
                                    <td className="py-2 px-2">{inc.eventType}</td>
                                    <td className="py-2 px-2">{inc.escalationTarget}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};
