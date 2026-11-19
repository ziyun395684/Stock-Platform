export interface StrategyVersion {
  id: string
  strategy_id: string
  version: string
  status: string
  config: Record<string, unknown>
  data_version?: string | null
  factor_version_ids?: string[] | null
  model_version?: string | null
}

export interface Strategy {
  id: string
  strategy_id: string
  name: string
  status: string
  description?: string | null
  parent_strategy_id?: string | null
  versions: StrategyVersion[]
}

export interface Approval {
  id: string
  target_type: string
  target_id: string
  decision: string
  reviewer: string
  comment?: string | null
  report_artifact_ids: string[]
}

export interface Experiment {
  id: string
  experiment_id: string
  objective: string
  status: string
  config: Record<string, unknown>
  strategy_version_id?: string | null
  created_by?: string | null
}

export interface FactorVersion {
  id: string
  version: string
  formula: string
  input_columns: string[]
  parameters: Record<string, unknown>
  time_boundary?: string | null
  code_artifact_id?: string | null
  created_at?: string | null
}

export interface Factor {
  id: string
  factor_id: string
  name: string
  status: string
  economic_hypothesis?: string | null
  versions: FactorVersion[]
}

export interface DependencyStatus {
  configured: boolean
  reachable: boolean
  detail: string
}

export interface Dependencies {
  database: DependencyStatus
  redis: DependencyStatus
}

export interface ExperimentRun {
  id: string
  experiment_id: string
  run_key: string
  status: string
  task_id?: string | null
  progress: number
  error_message?: string | null
  artifacts: Artifact[]
}

export interface Artifact {
  id: string
  artifact_type: string
  relative_path: string
  sha256?: string | null
  size_bytes?: number | null
}

export interface AgentSession {
  id: string
  objective: string
  status: string
  created_by?: string | null
  plan?: Record<string, unknown> | null
}

export interface AgentSessionState {
  session_id: string
  session: AgentSession | null
  tasks: AgentTask[]
  summary: { task_count: number; completed: number; failed: number; statuses: Record<string, number> }
  suggestions: Array<{ action: string; task_id?: string; reason: string }>
}

export interface AgentFeedback {
  session_id: string
  outcome: string
  completed: number
  failed: number
  runs: Array<{ task_id: string; task_type: string; run_id?: string; status: string; error?: string | null }>
  metrics: Record<string, number | null>
  suggestions: Array<{ action: string; reason: string }>
}

export interface AgentNextPlan {
  session_id: string
  parent_session_id: string
  feedback: AgentFeedback
  plan: ResearchPlan
  requires_confirmation: boolean
}

export interface ExperimentComparisonRow {
  id: string
  experiment_id: string
  objective: string
  runs: number
  completed_runs: number
  failed_runs: number
  metrics: Record<string, number | null>
}

export interface ExperimentComparison {
  experiments: ExperimentComparisonRow[]
  comparison: { best_return: string | null; lowest_drawdown: string | null; best_sharpe: string | null }
}

export interface AgentTask {
  id: string
  session_id: string
  task_type: string
  tool_name?: string | null
  status: string
  payload: Record<string, unknown>
}

export interface ResearchGoal {
  objective: string
  baseline_strategy_id?: string | null
  constraints: string[]
  budget: number
}

export interface ResearchPlan {
  goal: ResearchGoal
  tasks: string[]
  estimated_runs: number
  requires_approval: boolean
}

export interface ExperimentDesign {
  valid: boolean
  experiments: Array<{ task: string; training_window: number; seed: number; cost_rate: number }>
  estimated_runs: number
  budget: number
  reason?: string | null
  rationale: string[]
}

export interface ResearchReview {
  passed: boolean
  summary: string
  findings: Array<{ check: string; status: 'pass' | 'warn' | 'fail'; evidence: string; recommendation?: string | null }>
}

export interface Job {
  id: string
  experiment_run_id: string
  task_id?: string | null
  status: string
  progress: number
  current_epoch?: number | null
  total_epochs?: number | null
  experiment_id?: string | null
  run_key?: string | null
  run_status?: string | null
  started_at?: string | null
  finished_at?: string | null
  error_type?: string | null
  error_message?: string | null
  artifacts: Artifact[]
}

export interface ReportSeries {
  name: string
  points: Array<{ x: string; y: number }>
}

export interface ReportSummary {
  artifact_type: string
  relative_path: string
  kind: string
  metrics: Record<string, number>
  preview: string
  columns: string[]
  rows: string[][]
  series: ReportSeries[]
}

export interface CandidateReport {
  report_type: string
  experiment_id: string
  experiment_objective: string
  selection_policy: string
  total_runs: number
  completed_observations: number
  coverage: { windows: string[]; seeds: string[]; cost_rates: string[]; market_regimes: string[] }
  checks: Record<string, boolean>
  overall: { count: number; mean: number | null; std: number | null; positive_ratio: number | null }
  groups: Array<{ window: string; seed: string; cost_rate: string; regime: string; metrics: Record<string, { count: number; mean: number | null; std: number | null; positive_ratio: number | null }> }>
  recommendation: { status: string; eligible: boolean; reason: string | null }
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    const message = body?.error?.message || body?.detail || `API 请求失败 (${response.status})`
    throw new Error(message)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; service: string; timestamp: string }>('/health'),
  dependencies: () => request<Dependencies>('/dependencies'),
  strategies: () => request<Strategy[]>('/strategies'),
  createStrategy: (payload: { strategy_id: string; name: string; description?: string }) =>
    request<Strategy>('/strategies', { method: 'POST', body: JSON.stringify(payload) }),
  cloneStrategy: (strategyId: string, payload: { strategy_id: string; name: string; description?: string }) =>
    request<Strategy>(`/strategies/${strategyId}/clone`, { method: 'POST', body: JSON.stringify(payload) }),
  transitionStrategy: (strategyId: string, status: string, reason?: string) =>
    request<Strategy>(`/strategies/${strategyId}/transition`, { method: 'POST', body: JSON.stringify({ status, reason }) }),
  strategyApprovals: (strategyId: string) => request<Approval[]>(`/strategies/${strategyId}/approvals`),
  submitApproval: (payload: { target_type: 'strategy' | 'strategy_version' | 'experiment'; target_id: string; decision: 'approved' | 'rejected' | 'freeze_requested'; reviewer: string; comment?: string }) =>
    request<Approval>('/approvals', { method: 'POST', body: JSON.stringify(payload) }),
  experiments: () => request<Experiment[]>('/experiments'),
  factors: () => request<Array<Factor | Artifact>>('/factors'),
  jobs: (status?: string) => request<Job[]>(`/jobs${status ? `?status=${encodeURIComponent(status)}` : ''}`),
  cancelJob: (jobId: string) => request<Job>(`/jobs/${jobId}/cancel`, { method: 'POST' }),
  retryJob: (jobId: string) => request<ExperimentRun>(`/jobs/${jobId}/retry`, { method: 'POST' }),
  jobReport: (jobId: string) => request<{ job: Job; summaries: ReportSummary[] }>(`/jobs/${jobId}/report`),
  candidateReport: (experimentId: string) => request<CandidateReport>(`/reports/candidate/${experimentId}`),
  runs: (experimentId: string) => request<ExperimentRun[]>(`/experiments/${experimentId}/runs`),
  createRun: (experimentId: string, payload: { run_key: string; task_type: string; config: Record<string, unknown> }) => request<ExperimentRun>(`/experiments/${experimentId}/runs`, { method: 'POST', body: JSON.stringify(payload) }),
  artifactPreview: (id: string) => request<{ id: string; artifact_type: string; relative_path: string; content: string; columns: string[]; rows: string[][] }>(`/artifacts/${id}/preview`),
  createExperiment: (payload: { experiment_id: string; objective: string; config: Record<string, unknown> }) =>
    request<Experiment>('/experiments', { method: 'POST', body: JSON.stringify(payload) }),
  agentSessions: () => request<AgentSession[]>('/agents/sessions'),
  agentTasks: (sessionId: string) => request<AgentTask[]>(`/agents/sessions/${sessionId}/tasks`),
  agentSummary: (sessionId: string) => request<{ task_count: number; completed: number; failed: number; statuses: Record<string, number> }>(`/agents/sessions/${sessionId}/summary`),
  agentState: (sessionId: string) => request<AgentSessionState>(`/agents/sessions/${sessionId}/state`),
  agentFeedback: (sessionId: string) => request<AgentFeedback>(`/agents/sessions/${sessionId}/feedback`),
  agentNextPlan: (sessionId: string) => request<AgentNextPlan>(`/agents/sessions/${sessionId}/next-plan`, { method: 'POST' }),
  compareExperiments: (ids: string[]) => request<ExperimentComparison>(`/experiments/compare?ids=${encodeURIComponent(ids.join(','))}`),
  agentMarkdownReport: async (sessionId: string) => {
    const response = await fetch(`${API_BASE}/agents/sessions/${sessionId}/report/markdown`)
    if (!response.ok) throw new Error('研究报告导出失败')
    return response.blob()
  },
  submitAgentTask: (taskId: string) => request<{ agent_task_id: string; task_type: string; run_id: string; celery_task_id: string | null; status: string; idempotent?: boolean }>(`/agents/tasks/${taskId}/submit`, { method: 'POST' }),
  validatePlan: (plan: ResearchPlan) => request<{ valid: boolean; estimated_runs: number; budget: number; requires_approval: boolean; reason: string | null }>('/agents/plan/validate', { method: 'POST', body: JSON.stringify(plan) }),
  draftPlan: (goal: ResearchGoal) => request<ResearchPlan>('/agents/plan/draft', { method: 'POST', body: JSON.stringify(goal) }),
  designPlan: (payload: { tasks: string[]; training_windows: number[]; seeds: number[]; cost_rates: number[]; max_experiments: number; budget: number }) => request<ExperimentDesign>('/agents/plan/design', { method: 'POST', body: JSON.stringify(payload) }),
  reviewPlan: (payload: { config: Record<string, unknown>; metrics: Record<string, unknown> }) => request<ResearchReview>('/agents/review', { method: 'POST', body: JSON.stringify(payload) }),
  confirmPlan: (plan: ResearchPlan) => request<{ status: string; objective: string; tasks: string[]; estimated_runs: number; requires_approval: boolean }>('/agents/plan/confirm', { method: 'POST', body: JSON.stringify(plan) }),
  executePlan: (plan: ResearchPlan) => request<{ status: string; session: AgentSession; queued: Array<{ task_id: string; task_type: string; experiment: Experiment; run: ExperimentRun }> }>('/agents/plan/execute', { method: 'POST', body: JSON.stringify(plan) }),
  cancelPlan: (plan: ResearchPlan) => request<{ status: string; objective: string; tasks: string[]; estimated_runs: number }>('/agents/plan/cancel', { method: 'POST', body: JSON.stringify(plan) }),
}
