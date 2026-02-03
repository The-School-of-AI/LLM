export type RunState =
  | 'RUNNING'
  | 'PAUSED'
  | 'HALTED'
  | 'DEGRADED'
  | 'COMPLETED';

export type TrainingPhase =
  | 'pretrain'
  | 'post-train'
  | 'SFT'
  | 'RL';

export interface RunStatus {
  runId: string;
  modelName: string;
  phase: TrainingPhase;
  state: RunState;
  currentStep: number;
  tokensProcessed: number;
  wallClockRuntimeSeconds: number;
  lastUpdated: string; // ISO timestamp
  source: string; // e.g. "Team 12 Cluster", "Auto-Recover"
}

export type RoutingHealth = 'OK' | 'DEGRADED' | 'UNKNOWN';

export interface DataPoint {
  timestamp: string; // ISO timestamp
  value: number;
}

export interface LiveMetrics {
  loss: DataPoint[];
  throughput: DataPoint[]; // tokens/sec
  gpuUtilization: DataPoint[]; // %
  gpuMemory: DataPoint[]; // %
  routingHealth: RoutingHealth;
  lastUpdated: string;
}

export interface CheckpointStatus {
  lastCheckpointStep: number;
  checkpointTimestamp: string;
  checkpointId: string;
  hasOptimizerState: boolean;
  growthPhase: string | null;
  loraLockSummary: string | null;
  lastUpdated: string;
}

export type IncidentSeverity =
  | 'SEV-1'
  | 'SEV-2'
  | 'SEV-3';

export type IncidentStatus =
  | 'OPEN'
  | 'PAUSED'
  | 'RESOLVED';

export interface Incident {
  id: string; // Unique ID for the incident
  severity: IncidentSeverity;
  eventType: string; // e.g., 'pause', 'halt', 'degradation'
  timestamp: string;
  status: IncidentStatus;
  escalationTarget: string;
  source: string; // e.g. "Watchdog", "SRE"
}

export type DriftStatus =
  | 'OK'
  | 'WARNING';

export interface CostStatus {
  currentBurnRate: number; // $/hr
  expectedSpend: number;
  actualSpend: number;
  driftStatus: DriftStatus;
  haltProximity?: number; // 0-100
  lastUpdated: string;
}

export interface DashboardData {
  runStatus: RunStatus;
  liveMetrics: LiveMetrics;
  checkpointStatus: CheckpointStatus;
  incidents: Incident[];
  costStatus: CostStatus;
}
