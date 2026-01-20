"""Pydantic models for API requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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

