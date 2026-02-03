import React from 'react';
import type { LiveMetrics, RoutingHealth, DataPoint } from '../../types/dashboard';
import { formatDateTime } from '../../utils/time';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Cpu, Server, Network } from 'lucide-react';
import clsx from 'clsx';
import './LiveSignals.css';

interface LiveSignalsProps {
    metrics: LiveMetrics;
}

const ChartCard: React.FC<{ title: string; icon: React.ReactNode; data: DataPoint[]; dataKey: string; color: string; unit?: string }> = ({ title, icon, data, dataKey, color, unit }) => {
    return (
        <div className="bg-surface border rounded p-4 flex flex-col h-64">
            <div className="flex items-center gap-2 mb-2 text-sm font-bold text-muted">
                {icon} {title}
            </div>
            <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                        <XAxis
                            dataKey="timestamp"
                            tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }}
                            tickFormatter={(val) => new Date(val).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        />
                        <YAxis
                            tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }}
                            width={40}
                            domain={['auto', 'auto']}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'var(--color-surface)', borderColor: 'var(--color-border)', fontSize: '12px' }}
                            itemStyle={{ color: 'var(--color-text)' }}
                            labelFormatter={(label) => new Date(label).toLocaleTimeString()}
                            formatter={(value: number | undefined) => [(value !== undefined ? value.toFixed(2) : '0.00') + (unit ? ` ${unit}` : ''), title]}
                        />
                        <Line
                            type="monotone"
                            dataKey={dataKey}
                            stroke={color}
                            strokeWidth={2}
                            dot={false}
                            isAnimationActive={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

const RoutingHealthBadge: React.FC<{ status: RoutingHealth }> = ({ status }) => {
    const colorClass = clsx({
        'bg-success': status === 'OK',
        'bg-warning': status === 'UNKNOWN',
        'bg-danger': status === 'DEGRADED',
    });
    return (
        <div className={`routing-health-badge ${colorClass} text-sm font-bold flex items-center gap-2 px-3 py-1 rounded-full`}>
            <Network size={16} />
            Routing: {status}
        </div>
    );
}

export const LiveSignals: React.FC<LiveSignalsProps> = ({ metrics }) => {
    return (
        <div className="mb-6">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold">Live Signals</h2>
                <div className="flex items-center gap-4">
                    <RoutingHealthBadge status={metrics.routingHealth} />
                    <div className="text-sm text-muted">Last updated: {formatDateTime(metrics.lastUpdated)}</div>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ChartCard
                    title="Loss"
                    icon={<Activity size={16} />}
                    data={metrics.loss}
                    dataKey="value"
                    color="var(--color-primary)"
                />
                <ChartCard
                    title="Throughput (tokens/s)"
                    icon={<Server size={16} />}
                    data={metrics.throughput}
                    dataKey="value"
                    color="var(--color-success)"
                />
                <ChartCard
                    title="GPU Utilization (%)"
                    icon={<Cpu size={16} />}
                    data={metrics.gpuUtilization}
                    dataKey="value"
                    color="var(--color-warning)"
                    unit="%"
                />
                <ChartCard
                    title="GPU Memory (%)"
                    icon={<Database size={16} />} // Using Database icon for memory as 'Memory' component icon might not exist or verify later. Lucide has 'MemoryStick' or 'Cpu' etc. Database is close enough.
                    data={metrics.gpuMemory}
                    dataKey="value"
                    color="var(--color-danger)"
                    unit="%"
                />
            </div>
        </div>
    );
};
import { Database } from 'lucide-react';
