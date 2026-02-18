import React from 'react';
import {
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import type { LiveMetrics, DataPoint } from '../../types/dashboard';
import { Activity, Zap, Server, Cpu, AlertCircle, CheckCircle } from 'lucide-react';
import clsx from 'clsx';

interface LiveSignalsProps {
    metrics: LiveMetrics;
}

// Custom Tooltip for that premium feel
interface CustomTooltipProps {
    active?: boolean;
    payload?: { value: number }[];
    label?: string;
    unit?: string;
}

const CustomTooltip = ({ active, payload, label, unit }: CustomTooltipProps) => {
    if (active && payload && payload.length) {
        return (
            <div className="glass-panel p-3 border border-border/50 shadow-xl">
                <p className="text-xs text-muted mb-1">{label ? new Date(label).toLocaleTimeString() : ''}</p>
                <p className="text-sm font-bold text-white">
                    {payload[0].value.toFixed(2)}
                    <span className="text-dim ml-1">{unit}</span>
                </p>
            </div>
        );
    }
    return null;
};

const ChartCard: React.FC<{
    title: string;
    icon: React.ReactNode;
    data: DataPoint[];
    color: string;
    unit?: string;
    gradientId: string;
}> = ({ title, icon, data, color, unit, gradientId }) => {
    return (
        <div className="glass-panel p-5 relative overflow-hidden group hover:border-border-light transition-colors">
            <div className="flex justify-between items-center mb-4 relative z-10">
                <h3 className="text-sm font-bold text-muted uppercase tracking-wider flex items-center gap-2">
                    {icon}
                    {title}
                </h3>
                <div className="text-xl font-bold font-mono text-white">
                    {data[data.length - 1]?.value.toFixed(2)}
                    <span className="text-xs text-dim ml-1">{unit}</span>
                </div>
            </div>

            <div className="h-48 w-full -mx-2">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={color} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis
                            dataKey="timestamp"
                            hide={true}
                        />
                        <YAxis
                            hide={true}
                            domain={['auto', 'auto']}
                        />
                        <Tooltip content={<CustomTooltip unit={unit} />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1 }} />
                        <Area
                            type="monotone"
                            dataKey="value"
                            stroke={color}
                            strokeWidth={2}
                            fillOpacity={1}
                            fill={`url(#${gradientId})`}
                            isAnimationActive={false}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export const LiveSignals: React.FC<LiveSignalsProps> = ({ metrics }) => {
    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold flex items-center gap-2 text-white">
                    <Zap className="text-yellow-400" size={20} fill="currentColor" />
                    Live Signals
                </h2>
                <div className={clsx("flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold border",
                    metrics.routingHealth === 'OK' ? "bg-success/10 text-success border-success/20" : "bg-danger/10 text-danger border-danger/20"
                )}>
                    {metrics.routingHealth === 'OK' ? <CheckCircle size={12} /> : <AlertCircle size={12} />}
                    Routing: {metrics.routingHealth}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <ChartCard
                    title="Training Loss"
                    icon={<Activity size={16} />}
                    data={metrics.loss}
                    color="#ec4899" // Pink 500
                    gradientId="gradLoss"
                />
                <ChartCard
                    title="Throughput"
                    icon={<Zap size={16} />}
                    data={metrics.throughput}
                    color="#f59e0b" // Amber 500
                    unit="tok/s"
                    gradientId="gradThroughput"
                />
                <ChartCard
                    title="GPU Utilization"
                    icon={<Cpu size={16} />}
                    data={metrics.gpuUtilization}
                    color="#3b82f6" // Blue 500
                    unit="%"
                    gradientId="gradGpu"
                />
                <ChartCard
                    title="HBM Usage"
                    icon={<Server size={16} />}
                    data={metrics.gpuMemory}
                    color="#8b5cf6" // Violet 500
                    unit="%"
                    gradientId="gradMem"
                />
            </div>
        </div>
    );
};
