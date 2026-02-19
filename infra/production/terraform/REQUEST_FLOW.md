# Request Flow: API Gateway → EC2 Docker Container

## Complete Request Flow

When you send a query/task request to API Gateway, here's exactly what happens:

```
1. Frontend/Client
   │
   │ POST /api/v1/tasks
   │ WebSocket: wss://api-gateway-url/ws
   │
   ▼
2. API Gateway (HTTP API)
   │
   │ Routes ALL requests to ALB
   │ (via HTTP_PROXY integration)
   │
   ▼
3. Application Load Balancer (ALB)
   │
   │ Path-based routing:
   │ - /ws, /ws/* → Backend Target Group (port 8000)
   │ - /api/* → Backend Target Group (port 8000)
   │ - /health → Backend Target Group (port 8000)
   │ - /* → Frontend Target Group (port 3000)
   │
   ▼
4. EC2 Instance (Auto Scaling Group)
   │
   │ Docker Compose running:
   │ - Backend container (FastAPI) :8000
   │ - Frontend container (Next.js) :3000
   │
   ▼
5. Backend FastAPI Application
   │
   │ Receives request on port 8000
   │
   │ For REST API:
   │ - /api/v1/tasks → tasks.py router
   │ - /api/v1/projects → projects.py router
   │ - /api/v1/models → main.py endpoint
   │
   │ For WebSocket:
   │ - /ws → websocket handler
   │ - Creates agent session
   │ - Runs OpenBrowser task
   │ - Streams updates back via WebSocket
   │
   ▼
6. OpenBrowser Framework
   │
   │ Executes browser automation task
   │ - Opens browser (Playwright/Chromium)
   │ - Performs actions
   │ - Captures screenshots
   │ - Returns results
   │
   ▼
7. Response flows back
   │
   │ Backend → ALB → API Gateway → Frontend/Client
   │
   │ Real-time updates via WebSocket:
   │ - step_update
   │ - screenshot
   │ - output
   │ - task_completed
```

## Example: Sending a Task

### Via WebSocket (Real-time)

```javascript
// Frontend connects to API Gateway WebSocket endpoint
const ws = new WebSocket('wss://<api-gateway-url>/ws');

// Send task request
ws.send(JSON.stringify({
  type: 'start_task',
  data: {
    task: 'Search for OpenBrowser on GitHub',
    agent_type: 'code',
    max_steps: 50,
    use_vision: true,
    llm_model: 'gemini-2.5-flash'
  }
}));

// Receive real-time updates
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  // message.type: 'task_started', 'step_update', 'screenshot', 'task_completed'
};
```

**Flow:**
1. WebSocket connection: `wss://api-gateway-url/ws`
2. API Gateway → ALB → Backend container (port 8000)
3. Backend creates agent session
4. Task runs on EC2 instance (browser automation)
5. Updates streamed back via WebSocket

### Via REST API

```javascript
// Frontend calls API Gateway REST endpoint
const response = await fetch('https://<api-gateway-url>/api/v1/tasks', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    task: 'Search for OpenBrowser on GitHub',
    agent_type: 'code',
    max_steps: 50
  })
});
```

**Flow:**
1. REST request: `https://api-gateway-url/api/v1/tasks`
2. API Gateway → ALB → Backend container (port 8000)
3. Backend processes request
4. Response returned

## Where Tasks Actually Run

**YES - Tasks run on the dockerized EC2 instance!**

- The EC2 instance runs Docker Compose
- Docker Compose starts the backend container
- The backend container runs FastAPI + OpenBrowser
- When you send a task, it executes on that EC2 instance
- The browser automation happens on that EC2 instance
- Screenshots are captured on that EC2 instance
- Results are computed on that EC2 instance

## Multi-Instance Behavior

If you have multiple EC2 instances (Auto Scaling):

- ALB distributes requests across instances (round-robin)
- Each instance can handle multiple concurrent tasks
- WebSocket connections are sticky (same instance for duration)
- Tasks run independently on each instance

## Important Notes

1. **WebSocket Support**: API Gateway HTTP API supports WebSocket connections, so real-time updates work through the API Gateway.

2. **Health Checks**: ALB continuously checks backend health at `/health`. Unhealthy instances are removed from rotation.

3. **Scaling**: If traffic increases, Auto Scaling Group can add more EC2 instances automatically.

4. **State**: Each EC2 instance maintains its own state. For persistent storage, consider adding a database or Redis.

5. **VNC**: If enabled, VNC connections for live browser viewing also go through the same infrastructure.

## Testing the Flow

1. Deploy infrastructure: `make terraform-apply-prod`
2. Get API Gateway URL: `make prod-get-url`
3. Test health endpoint:
   ```bash
   curl https://<api-gateway-url>/health
   ```
4. Test WebSocket connection:
   ```javascript
   const ws = new WebSocket('wss://<api-gateway-url>/ws');
   ```

## Troubleshooting

If tasks aren't running:

1. **Check ALB target health**: Ensure backend target group shows healthy instances
2. **Check EC2 logs**: SSH to instance and check Docker logs
   ```bash
   docker compose -f docker-compose.prod.yml logs backend
   ```
3. **Check API Gateway logs**: CloudWatch logs at `/aws/apigateway/openbrowser`
4. **Verify routing**: Ensure ALB listener rules are correct (check AWS Console)
