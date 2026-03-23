import { API_BASE_URL } from "@/lib/config";
import type {
  BackendJobExecution,
  BackendScheduledJob,
  JobExecution,
  ScheduledJob,
} from "@/types";

function authHeaders(token: string | null): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function mapJob(j: BackendScheduledJob): ScheduledJob {
  return {
    id: j.id,
    title: j.title,
    taskDescription: j.task_description,
    workflowId: j.workflow_id,
    authProfileId: j.auth_profile_id,
    scheduleExpression: j.schedule_expression,
    scheduleTimezone: j.schedule_timezone,
    status: j.status as ScheduledJob["status"],
    lastRunAt: j.last_run_at,
    nextRunAt: j.next_run_at,
    createdAt: j.created_at,
    updatedAt: j.updated_at,
  };
}

function mapExecution(e: BackendJobExecution): JobExecution {
  return {
    id: e.id,
    status: e.status as JobExecution["status"],
    startedAt: e.started_at,
    completedAt: e.completed_at,
    errorMessage: e.error_message,
    taskId: e.task_id,
    createdAt: e.created_at,
  };
}

export async function fetchScheduledJobs(token: string | null): Promise<ScheduledJob[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/schedules`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to fetch schedules: ${await res.text()}`);
  const data = await res.json();
  return (data.jobs as BackendScheduledJob[]).map(mapJob);
}

export async function createScheduledJob(
  token: string | null,
  params: {
    title: string;
    task_description: string;
    schedule_expression: string;
    schedule_timezone?: string;
    auth_profile_id?: string;
  },
): Promise<ScheduledJob> {
  const res = await fetch(`${API_BASE_URL}/api/v1/schedules`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Failed to create schedule: ${await res.text()}`);
  return mapJob(await res.json());
}

export async function updateScheduledJob(
  token: string | null,
  jobId: string,
  params: { title?: string; schedule_expression?: string; schedule_timezone?: string; status?: string },
): Promise<ScheduledJob> {
  const res = await fetch(`${API_BASE_URL}/api/v1/schedules/${jobId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Failed to update schedule: ${await res.text()}`);
  return mapJob(await res.json());
}

export async function deleteScheduledJob(token: string | null, jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/v1/schedules/${jobId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to delete schedule: ${await res.text()}`);
}

export async function fetchExecutions(token: string | null, jobId: string): Promise<JobExecution[]> {
  const res = await fetch(`${API_BASE_URL}/api/v1/schedules/${jobId}/executions`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(`Failed to fetch executions: ${await res.text()}`);
  const data = await res.json();
  return (data.executions as BackendJobExecution[]).map(mapExecution);
}
