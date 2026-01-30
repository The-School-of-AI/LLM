import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { AlertCircle, CheckCircle, TrendingUp, Zap, Activity, Target, Server, Clock } from 'lucide-react';

// Simulated real-time data generator
const generateMetrics = (step) => {
  const noise = () => (Math.random() - 0.5) * 2;
  const progress = Math.min(1, step / 500);
  
  return {
    step,
    timestamp: new Date().toISOString(),
    junkToNull: Math.max(0, Math.min(100, 65 + progress * 10 + Math.sin(step / 30) * 3 + noise())),
    signalToNull: Math.max(0, Math.min(100, 8 - progress * 3 + noise() * 0.5)),
    boilerplateToNull: Math.max(0, Math.min(100, 50 + progress * 8 + noise())),
    entropy: Math.max(0, Math.min(1, 0.75 + progress * 0.15 + Math.sin(step / 100) * 0.03 + noise() * 0.01)),
    gini: Math.max(0, Math.min(1, 0.25 - progress * 0.1 + noise() * 0.02)),
    computeSavings: Math.max(0, 12 + progress * 5 + noise() * 0.5),
    stabilityScore: Math.max(0, Math.min(1, 0.7 + progress * 0.2 + noise() * 0.02)),
    loraReady: progress > 0.5 && Math.random() > 0.2,
    growthReady: progress > 0.7 && Math.random() > 0.6,
    deadExperts: Math.random() > 0.9 ? [Math.floor(Math.random() * 64)] : [],
    throughput: Math.floor(45000 + noise() * 2000),
    loss: Math.max(0.5, 4 - progress * 3 + noise() * 0.1),
  };
};

// Gauge component
const Gauge = ({ value, max = 100, label, target, color, inverse = false }) => {
  const percentage = (value / max) * 100;
  const isGood = inverse 
    ? value <= (target || max * 0.3)
    : value >= (target || max * 0.6);
  
  return (
    <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100">
      <div className="text-sm text-gray-500 mb-2">{label}</div>
      <div className="relative h-4 bg-gray-200 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full transition-all duration-500 ${isGood ? 'bg-green-500' : 'bg-amber-500'}`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
        {target && (
          <div 
            className="absolute top-0 h-full w-0.5 bg-gray-800"
            style={{ left: `${(target / max) * 100}%` }}
          />
        )}
      </div>
      <div className="flex justify-between items-center mt-2">
        <span className="text-2xl font-bold text-gray-800">
          {typeof value === 'number' ? value.toFixed(1) : value}
          {max === 100 ? '%' : ''}
        </span>
        <span className="text-xs text-gray-400">Target: {target}{max === 100 ? '%' : ''}</span>
      </div>
    </div>
  );
};

// Status badge component
const StatusBadge = ({ ready, label }) => (
  <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${ready ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
    {ready ? <CheckCircle size={18} /> : <Clock size={18} />}
    <span className="font-medium">{label}</span>
    <span className="text-sm">{ready ? 'Ready' : 'Pending'}</span>
  </div>
);

// Alert component
const Alert = ({ type, title, message }) => {
  const styles = {
    critical: 'bg-red-50 border-red-200 text-red-800',
    warning: 'bg-amber-50 border-amber-200 text-amber-800',
    info: 'bg-blue-50 border-blue-200 text-blue-800',
  };
  
  return (
    <div className={`p-4 rounded-lg border ${styles[type]} flex items-start gap-3`}>
      <AlertCircle size={20} className="mt-0.5 flex-shrink-0" />
      <div>
        <div className="font-semibold">{title}</div>
        <div className="text-sm opacity-80">{message}</div>
      </div>
    </div>
  );
};

// Health gate indicator
const HealthGate = ({ name, passed }) => (
  <div className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
    <span className="text-sm text-gray-600">{name}</span>
    {passed ? (
      <CheckCircle size={18} className="text-green-500" />
    ) : (
      <AlertCircle size={18} className="text-red-500" />
    )}
  </div>
);

// Main Dashboard
export default function MoEDashboard() {
  const [metrics, setMetrics] = useState(generateMetrics(0));
  const [history, setHistory] = useState([]);
  const [step, setStep] = useState(0);
  const [refreshRate, setRefreshRate] = useState(2);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setStep(s => s + 1);
      const newMetrics = generateMetrics(step + 1);
      setMetrics(newMetrics);
      setHistory(prev => [...prev.slice(-99), {
        step: step + 1,
        junkToNull: newMetrics.junkToNull,
        signalToNull: newMetrics.signalToNull,
        entropy: newMetrics.entropy * 100,
        gini: newMetrics.gini * 100,
        loss: newMetrics.loss,
      }]);
    }, refreshRate * 1000);
    
    return () => clearInterval(interval);
  }, [step, refreshRate]);

  // Calculate health gates
  const healthGates = {
    'Junk → Null ≥ 60%': metrics.junkToNull >= 60,
    'Junk → Null ≤ 80%': metrics.junkToNull <= 80,
    'Signal → Null ≤ 10%': metrics.signalToNull <= 10,
    'Entropy ≥ 0.70': metrics.entropy >= 0.70,
    'Gini ≤ 0.50': metrics.gini <= 0.50,
    'No Dead Experts': metrics.deadExperts.length === 0,
  };
  
  const allGatesPass = Object.values(healthGates).every(v => v);
  
  // Generate alerts
  const alerts = [];
  if (metrics.signalToNull > 10) {
    alerts.push({ type: 'warning', title: 'High Signal Leakage', message: `${metrics.signalToNull.toFixed(1)}% signal tokens routing to null (target: <10%)` });
  }
  if (metrics.entropy < 0.5) {
    alerts.push({ type: 'critical', title: 'Entropy Collapse', message: 'Router entropy critically low - possible expert collapse' });
  }
  if (metrics.deadExperts.length > 0) {
    alerts.push({ type: 'warning', title: 'Dead Experts', message: `Expert(s) ${metrics.deadExperts.join(', ')} have <1% utilization` });
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-6 mb-6 text-white shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-3">
              <Target size={28} />
              Team 7 - MoE Routing Dashboard
            </h1>
            <p className="text-indigo-200 mt-1">Real-time monitoring for null expert routing and MoE health</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-indigo-200 text-sm">Training Step</div>
              <div className="text-2xl font-bold">{metrics.step.toLocaleString()}</div>
            </div>
            <div className={`px-4 py-2 rounded-lg ${allGatesPass ? 'bg-green-500' : 'bg-red-500'}`}>
              {allGatesPass ? '✓ Healthy' : '✗ Issues'}
            </div>
          </div>
        </div>
      </div>

      {/* Status Badges */}
      <div className="flex gap-4 mb-6 flex-wrap">
        <StatusBadge ready={allGatesPass} label="All Gates" />
        <StatusBadge ready={metrics.loraReady} label="LoRA" />
        <StatusBadge ready={metrics.growthReady} label="Growth" />
        <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-100 text-blue-700">
          <Zap size={18} />
          <span className="font-medium">Compute Savings</span>
          <span className="text-lg font-bold">{metrics.computeSavings.toFixed(1)}%</span>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-purple-100 text-purple-700">
          <Activity size={18} />
          <span className="font-medium">Throughput</span>
          <span className="text-lg font-bold">{(metrics.throughput / 1000).toFixed(1)}k tok/s</span>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-5 gap-4 mb-6">
        <Gauge 
          value={metrics.junkToNull} 
          label="🗑️ Junk → Null" 
          target={70}
          color="blue"
        />
        <Gauge 
          value={metrics.signalToNull} 
          max={30}
          label="⚠️ Signal Leakage" 
          target={10}
          inverse={true}
        />
        <Gauge 
          value={metrics.entropy * 100} 
          label="📊 Routing Entropy" 
          target={70}
        />
        <Gauge 
          value={metrics.gini * 100} 
          label="⚖️ Gini (Balance)" 
          target={30}
          inverse={true}
        />
        <Gauge 
          value={metrics.stabilityScore * 100} 
          label="📈 Stability Score" 
          target={80}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* Null Routing Trends */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <TrendingUp size={20} className="text-indigo-500" />
            Null Routing Trends
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="step" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Area type="monotone" dataKey="junkToNull" name="Junk → Null" stroke="#3b82f6" fill="#93c5fd" fillOpacity={0.3} />
              <Area type="monotone" dataKey="signalToNull" name="Signal → Null" stroke="#ef4444" fill="#fca5a5" fillOpacity={0.3} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Routing Health */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Activity size={20} className="text-purple-500" />
            Routing Health
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="step" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="entropy" name="Entropy (%)" stroke="#8b5cf6" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="gini" name="Gini (%)" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bottom Section */}
      <div className="grid grid-cols-3 gap-6">
        {/* Health Gates */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <Server size={20} className="text-green-500" />
            Health Gates
          </h3>
          <div className="space-y-1">
            {Object.entries(healthGates).map(([name, passed]) => (
              <HealthGate key={name} name={name} passed={passed} />
            ))}
          </div>
        </div>

        {/* Alerts */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 col-span-2">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            <AlertCircle size={20} className="text-red-500" />
            Active Alerts
          </h3>
          {alerts.length === 0 ? (
            <div className="flex items-center gap-3 p-4 bg-green-50 rounded-lg text-green-700">
              <CheckCircle size={24} />
              <span className="font-medium">No active alerts - all systems healthy!</span>
            </div>
          ) : (
            <div className="space-y-3">
              {alerts.map((alert, i) => (
                <Alert key={i} {...alert} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="mt-6 text-center text-sm text-gray-400">
        Refresh Rate: {refreshRate}s | Model: 70B MoE-64 | Experts: 64 routed + 4 shared + 2 null
      </div>
    </div>
  );
}
