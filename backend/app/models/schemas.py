"""Pydantic models for API requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any

import re

from pydantic import BaseModel, Field, field_validator


class AgentType(str, Enum):
    """Type of agent to use."""
    BROWSER = "browser"  # Standard Agent with browser actions
    CODE = "code"  # CodeAgent with Python code execution


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageRole(str, Enum):
    """Role of a message in the conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# Request Models

class CreateTaskRequest(BaseModel):
    """Request to create a new task."""
    task: str = Field(..., description="The task description", min_length=1)
    agent_type: AgentType = Field(default=AgentType.CODE, description="Type of agent to use")
    max_steps: int = Field(default=50, ge=1, le=200, description="Maximum steps")
    use_vision: bool = Field(default=True, description="Enable vision/screenshots")
    llm_model: str | None = Field(default=None, description="LLM model to use")
    project_id: str | None = Field(default=None, description="Project to associate task with")
    use_current_browser: bool = Field(default=False, description="Use current browser via Chrome extension")
    conversation_id: str | None = Field(default=None, description="Existing conversation to append messages to")
    auth_profile_id: str | None = Field(default=None, description="Auth profile to load into browser session")


class CreateProjectRequest(BaseModel):
    """Request to create a new project."""
    name: str = Field(..., description="Project name", min_length=1, max_length=100)
    description: str | None = Field(default=None, description="Project description")


class UpdateProjectRequest(BaseModel):
    """Request to update a project."""
    name: str | None = Field(default=None, description="New project name")
    description: str | None = Field(default=None, description="New project description")


# Response Models

class TaskMessage(BaseModel):
    """A message in the task conversation."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: dict[str, Any] | None = None


class TaskStep(BaseModel):
    """A single step in task execution."""
    step_number: int
    action: str | None = None
    code: str | None = None
    output: str | None = None
    error: str | None = None
    screenshot_url: str | None = None
    timestamp: datetime
    duration_ms: int | None = None


class TaskResponse(BaseModel):
    """Response containing task details."""
    id: str
    task: str
    status: TaskStatus
    agent_type: AgentType
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    result: str | None = None
    success: bool | None = None
    steps: list[TaskStep] = Field(default_factory=list)
    messages: list[TaskMessage] = Field(default_factory=list)
    project_id: str | None = None
    error: str | None = None


class TaskListItem(BaseModel):
    """Task item for list views."""
    id: str
    task: str
    status: TaskStatus
    agent_type: AgentType
    created_at: datetime
    project_id: str | None = None
    preview: str | None = None  # First 100 chars of result


class ProjectResponse(BaseModel):
    """Response containing project details."""
    id: str
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime
    task_count: int = 0


class ProjectListResponse(BaseModel):
    """Response containing list of projects."""
    projects: list[ProjectResponse]
    total: int


class TaskListResponse(BaseModel):
    """Response containing list of tasks."""
    tasks: list[TaskListItem]
    total: int
    page: int
    page_size: int


# Auth profile models

class StartAuthSessionRequest(BaseModel):
    """Request to start a browser session for auth capture."""
    domain: str = Field(..., description="Domain to navigate to for login", min_length=1, max_length=500)
    label: str = Field(..., description="Label for this auth profile", min_length=1, max_length=100)


class SaveAuthProfileRequest(BaseModel):
    """Request to capture auth state from active browser session."""
    task_id: str = Field(..., description="Task ID of the active browser session")
    domain: str = Field(..., description="Domain to capture cookies for")
    label: str = Field(..., description="Label for this auth profile", min_length=1, max_length=100)


class AuthProfileResponse(BaseModel):
    """Response containing auth profile details."""
    id: str
    domain: str
    label: str
    status: str
    last_verified_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class AuthProfileListResponse(BaseModel):
    """Response containing list of auth profiles."""
    profiles: list[AuthProfileResponse]


class UpdateAuthProfileRequest(BaseModel):
    """Request to update auth profile label."""
    label: str = Field(..., min_length=1, max_length=100)


# Scheduled job models

_CRON_FIELD_RE = re.compile(
    r"^[0-9*/,\-?LW#]+$"
)


def _validate_cron_expression(value: str) -> str:
    """Validate a basic 6-field EventBridge cron expression (min hr dom mon dow yr)."""
    fields = value.strip().split()
    if len(fields) != 6:
        raise ValueError(
            "Cron expression must have exactly 6 fields: "
            "minute hour day-of-month month day-of-week year"
        )
    for i, field in enumerate(fields):
        if not _CRON_FIELD_RE.match(field):
            raise ValueError(
                f"Invalid characters in cron field {i + 1}: '{field}'"
            )
    return value.strip()


class CreateScheduledJobRequest(BaseModel):
    """Request to create a scheduled job (starts test run)."""
    title: str = Field(..., min_length=1, max_length=200)
    task_description: str = Field(..., min_length=1, max_length=50000)
    schedule_expression: str = Field(..., description="Cron expression for EventBridge (6 fields)")
    schedule_timezone: str = Field(default="UTC", max_length=50)
    auth_profile_id: str | None = Field(default=None, description="Optional auth profile for the job")

    @field_validator("schedule_expression")
    @classmethod
    def check_cron(cls, v: str) -> str:
        return _validate_cron_expression(v)


class UpdateScheduledJobRequest(BaseModel):
    """Request to update a scheduled job."""
    title: str | None = Field(default=None, min_length=1, max_length=200)
    schedule_expression: str | None = Field(default=None)
    schedule_timezone: str | None = Field(default=None, max_length=50)
    status: str | None = Field(default=None, description="active | paused")

    @field_validator("schedule_expression")
    @classmethod
    def check_cron(cls, v: str | None) -> str | None:
        if v is not None:
            return _validate_cron_expression(v)
        return v


class JobExecutionResponse(BaseModel):
    """Response for a single job execution."""
    id: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    task_id: str | None = None
    created_at: datetime


class ScheduledJobResponse(BaseModel):
    """Response containing scheduled job details."""
    id: str
    title: str
    task_description: str
    workflow_id: str | None = None
    auth_profile_id: str | None = None
    schedule_expression: str
    schedule_timezone: str
    status: str
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ScheduledJobListResponse(BaseModel):
    """Response containing list of scheduled jobs."""
    jobs: list[ScheduledJobResponse]


class ScheduledJobDetailResponse(BaseModel):
    """Response containing scheduled job + recent executions."""
    job: ScheduledJobResponse
    executions: list[JobExecutionResponse]


# WebSocket Message Models

class WSMessageType(str, Enum):
    """WebSocket message types."""
    # Client -> Server
    START_TASK = "start_task"
    CANCEL_TASK = "cancel_task"
    PAUSE_TASK = "pause_task"
    RESUME_TASK = "resume_task"
    REQUEST_VNC = "request_vnc"  # Request VNC connection info
    
    # Server -> Client
    TASK_STARTED = "task_started"
    STEP_UPDATE = "step_update"
    THINKING = "thinking"
    ACTION = "action"
    OUTPUT = "output"
    ERROR = "error"
    SCREENSHOT = "screenshot"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    LOG = "log"  # Backend terminal log messages
    VNC_INFO = "vnc_info"  # VNC connection information
    EXTENSION_STATUS = "extension_status"  # Chrome extension connection status


class WSMessage(BaseModel):
    """WebSocket message envelope."""
    type: WSMessageType
    task_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSStartTaskData(BaseModel):
    """Data for START_TASK message."""
    task: str
    agent_type: AgentType = AgentType.CODE
    max_steps: int = 50
    use_vision: bool = True
    llm_model: str | None = None
    use_current_browser: bool = False
    conversation_id: str | None = None


class WSStepUpdateData(BaseModel):
    """Data for STEP_UPDATE message."""
    step_number: int
    total_steps: int
    action: str | None = None
    code: str | None = None
    thinking: str | None = None
    memory: str | None = None
    next_goal: str | None = None


class WSOutputData(BaseModel):
    """Data for OUTPUT message."""
    content: str
    is_final: bool = False


class WSScreenshotData(BaseModel):
    """Data for SCREENSHOT message."""
    url: str | None = None
    base64: str | None = None
    step_number: int


class FileAttachment(BaseModel):
    """File attachment data."""
    name: str = Field(..., description="File name")
    content: str | None = Field(default=None, description="File content (text or base64)")
    url: str | None = Field(default=None, description="URL to download file")
    type: str | None = Field(default=None, description="File type (csv, json, text, code, image, etc.)")
    mime_type: str | None = Field(default=None, description="MIME type")
    size: int | None = Field(default=None, description="File size in bytes")


class WSTaskCompletedData(BaseModel):
    """Data for TASK_COMPLETED message."""
    result: str
    success: bool
    total_steps: int
    duration_seconds: float
    attachments: list[FileAttachment] = Field(default_factory=list)


class WSLogData(BaseModel):
    """Data for LOG message (backend terminal output)."""
    level: str = "info"  # info, warning, error, debug
    message: str
    source: str | None = None  # e.g., "openbrowser.code_use.service", "agent"
    step_number: int | None = None


class WSVncInfoData(BaseModel):
    """Data for VNC_INFO message (VNC connection details)."""
    vnc_url: str = Field(..., description="WebSocket URL for noVNC connection")
    password: str = Field(..., description="VNC password for authentication")
    width: int = Field(default=1280, description="Display width in pixels")
    height: int = Field(default=1024, description="Display height in pixels")
    display: str | None = Field(default=None, description="X11 display string (e.g., ':99')")


# Models API Response

class LLMModel(BaseModel):
    """Available LLM model information."""
    id: str = Field(..., description="Model identifier to use in API calls")
    name: str = Field(..., description="Human-readable model name")
    provider: str = Field(..., description="Provider name (google, openai, anthropic)")


class AvailableModelsResponse(BaseModel):
    """Response containing available LLM models based on configured API keys."""
    models: list[LLMModel] = Field(default_factory=list, description="List of available models")
    providers: list[str] = Field(default_factory=list, description="List of available providers")
    default_model: str | None = Field(default=None, description="Default model to use")


# Chat persistence API models

class ChatConversation(BaseModel):
    """Conversation list/detail record."""
    id: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None


class ChatMessage(BaseModel):
    """Persisted chat message."""
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ChatListResponse(BaseModel):
    """List conversations response."""
    conversations: list[ChatConversation]
    active_conversation_id: str | None = None


class ChatConversationResponse(BaseModel):
    """Conversation with messages response."""
    conversation: ChatConversation
    messages: list[ChatMessage]


class CreateChatRequest(BaseModel):
    """Create conversation request."""
    title: str | None = Field(default=None, max_length=200)


class RenameChatRequest(BaseModel):
    """Rename conversation request."""
    title: str = Field(..., min_length=1, max_length=200)


class SetActiveChatRequest(BaseModel):
    """Set active conversation request."""
    conversation_id: str | None = None
