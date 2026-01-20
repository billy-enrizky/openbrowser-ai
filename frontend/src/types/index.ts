// WebSocket Message Types
export type WSMessageType =
  | "start_task"
  | "cancel_task"
  | "pause_task"
  | "resume_task"
  | "request_vnc"
  | "task_started"
  | "step_update"
  | "thinking"
  | "action"
  | "output"
  | "error"
  | "screenshot"
  | "task_completed"
  | "task_failed"
  | "task_cancelled"
  | "log"
  | "vnc_info";  // VNC connection information

export interface WSMessage {
  type: WSMessageType;
  task_id?: string;
  data: Record<string, unknown>;
  timestamp?: string;
}

// Log entry from backend
export interface LogEntry {
  id: string;
  level: "info" | "warning" | "error" | "debug";
  message: string;
  source?: string;
  stepNumber?: number;
  timestamp: Date;
}

// VNC connection information
export interface VncInfo {
  vnc_url: string;
  password: string;
  width: number;
  height: number;
  display?: string;
}

// Browser viewer mode
export type BrowserViewerMode = "embedded" | "popup";

export type AgentType = "browser" | "code";

export type TaskStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface Task {
  id: string;
  task: string;
  status: TaskStatus;
  agentType: AgentType;
  createdAt: Date;
  updatedAt?: Date;
  completedAt?: Date;
  result?: string;
  success?: boolean;
  error?: string;
  steps: TaskStep[];
}

export interface TaskStep {
  stepNumber: number;
  action?: string;
  code?: string;
  thinking?: string;
  memory?: string;
  nextGoal?: string;
  output?: string;
  error?: string;
  screenshotUrl?: string;
  timestamp: Date;
  durationMs?: number;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  createdAt: Date;
  updatedAt: Date;
  taskCount: number;
}

// File attachment types
export type FileType = "csv" | "json" | "text" | "image" | "pdf" | "code" | "unknown";

export interface FileAttachment {
  id: string;
  name: string;
  type: FileType;
  mimeType?: string;
  size?: number;
  url?: string;           // URL to download/view the file
  content?: string;       // Base64 or text content for preview
  previewContent?: string; // Truncated content for preview
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  taskId?: string;
  metadata?: {
    stepNumber?: number;
    isThinking?: boolean;
    isError?: boolean;
    screenshot?: string;
    attachments?: FileAttachment[];
  };
}
