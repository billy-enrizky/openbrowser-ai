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

### Docker

Build and run from the repository root:

```bash
docker build -f backend/Dockerfile -t openbrowser-backend .
docker run -p 8000:8000 --env-file backend/.env openbrowser-backend
```

### Deploy to openbrowser.me

1. **Option A: Railway/Render**
   - Connect GitHub repo
   - Set environment variables
   - Deploy automatically

2. **Option B: VPS with Docker**
   - Set up Docker on VPS
   - Configure nginx reverse proxy
   - Point api.openbrowser.me to the server

3. **Option C: Cloudflare Workers + Durable Objects**
   - For serverless deployment
   - Requires adaptation for long-running tasks

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
