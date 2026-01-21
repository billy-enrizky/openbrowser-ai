# OpenBrowser Backend

Backend API for the OpenBrowser AI Chat Interface. A FastAPI-based server that provides WebSocket and REST APIs for real-time browser automation using the OpenBrowser framework.

## Features

- **WebSocket API** for real-time agent communication
- **REST API** for task and project management
- **Support for both Agent and CodeAgent** from openbrowser
- **Real-time streaming** of agent steps, outputs, and screenshots
- **Log streaming** to frontend for debugging

## Quick Start

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Installation

```bash
# Install dependencies
cd backend
uv pip install -e .

# Install openbrowser from local path
uv pip install -e /path/to/openbrowser-ai

# Copy environment file
cp env.example .env
# Edit .env with your API keys
```

### Running the Server

```bash
# Development mode
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| GET | `/api/v1/tasks` | List tasks |
| GET | `/api/v1/tasks/{task_id}` | Get task details |
| DELETE | `/api/v1/tasks/{task_id}` | Cancel task |
| GET | `/api/v1/projects` | List projects |
| POST | `/api/v1/projects` | Create project |
| GET | `/api/v1/projects/{project_id}` | Get project |
| PATCH | `/api/v1/projects/{project_id}` | Update project |
| DELETE | `/api/v1/projects/{project_id}` | Delete project |

### WebSocket API

Connect to `/ws` or `/ws/{client_id}` for real-time communication.

#### Client -> Server Messages

```json
{
  "type": "start_task",
  "data": {
    "task": "Search for OpenBrowser on GitHub",
    "agent_type": "code",
    "max_steps": 50,
    "use_vision": true
  }
}
```

```json
{
  "type": "cancel_task",
  "task_id": "uuid"
}
```

#### Server -> Client Messages

| Type | Description |
|------|-------------|
| `task_started` | Task has started |
| `step_update` | Agent step progress |
| `thinking` | Agent thinking/reasoning |
| `action` | Agent action being executed |
| `output` | Agent output/result |
| `screenshot` | Browser screenshot (base64) |
| `log` | Backend log message |
| `task_completed` | Task completed successfully |
| `task_failed` | Task failed with error |
| `task_cancelled` | Task was cancelled |

## Deployment

### Docker Compose (Recommended)

The backend requires VNC support for live browser viewing, which needs special Docker privileges. Use Docker Compose from the repository root:

```bash
# Development
docker-compose up --build

# Production
docker-compose -f docker-compose.yml up -d
```

This starts both the backend (port 8000) and frontend (port 3000) with proper VNC configuration.

### VNC Requirements

The backend uses VNC for live browser streaming, which requires:
- `shm_size: 2gb` - Shared memory for Chromium
- `seccomp:unconfined` - Security context for X11
- Xvfb, x11vnc, websockify - Display server and VNC tools

These requirements mean **PaaS platforms like Railway/Render are not compatible** with VNC mode.

### Production Deployment Options

1. **VPS with Docker Compose** (Recommended)
   - Use a cloud VM (AWS EC2, GCP Compute Engine, DigitalOcean Droplet, etc.)
   - Install Docker and Docker Compose
   - Clone the repository and run `docker-compose up -d`
   - Configure nginx reverse proxy with SSL
   - Point api.openbrowser.me to the server

2. **Headless Mode (No VNC)**
   - Set `VNC_ENABLED=false` in environment
   - Can deploy to Railway/Render without VNC live viewing
   - Screenshots still work, but no real-time browser streaming

## Architecture

```
backend/
  app/
    __init__.py
    main.py              # FastAPI app entry point
    api/
      __init__.py
      tasks.py           # Task REST endpoints
      projects.py        # Project REST endpoints
    core/
      __init__.py
      config.py          # Settings and configuration
    models/
      __init__.py
      schemas.py         # Pydantic models
    services/
      __init__.py
      agent_service.py   # Agent session management
      vnc_service.py     # VNC session management (Xvfb, x11vnc, websockify)
    websocket/
      __init__.py
      handler.py         # WebSocket message handling
```

## Environment Variables

See `env.example` for all available configuration options:

- `HOST` / `PORT` - Server binding
- `DEBUG` - Enable debug mode
- `CORS_ORIGINS` - Allowed origins (comma-separated)
- `DEFAULT_MAX_STEPS` - Default max agent steps
- `DEFAULT_AGENT_TYPE` - Default agent type (code/browser)
- `DEFAULT_LLM_MODEL` - Default LLM model
- `MAX_CONCURRENT_AGENTS` - Max concurrent agents
- `REDIS_URL` - Optional Redis for session persistence
- `GOOGLE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` - LLM API keys

### VNC Configuration

- `VNC_ENABLED` - Enable VNC live browser streaming (default: true)
- `VNC_WIDTH` - Browser viewport width (default: 1920)
- `VNC_HEIGHT` - Browser viewport height (default: 1080)
- `VNC_PASSWORD` - Optional VNC password for security
