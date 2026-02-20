# Request Flow: CloudFront -> ALB -> EC2 Docker Container

## Architecture Overview

CloudFront is the single entry point for all traffic. API Gateway exists in Terraform but is **completely bypassed** -- CloudFront routes directly to the ALB to avoid the 30-second integration timeout that kills persistent WebSocket connections.

## Complete Request Flow

```mermaid
graph LR
    Client[Client Browser] -->|HTTPS| CF[CloudFront]
    CF -->|"/api/*, /ws/*, /health"| ALB[ALB :80]
    CF -->|"/api/v1/vnc/*<br>persistent WS"| ALB
    CF -->|"/* static<br>CF Function URI rewrite"| S3[S3 Bucket]
    ALB --> FastAPI

    subgraph EC2["EC2 Private Subnet - Docker"]
        FastAPI["FastAPI :8000"]
        Chromium["Playwright/Chromium kiosk"]
        VNC["Xvfb + x11vnc + websockify"]
    end

    FastAPI --> RDS[(PostgreSQL RDS)]

    style Client fill:#4a90d9,color:#fff
    style CF fill:#7b68ee,color:#fff
    style ALB fill:#7b68ee,color:#fff
    style S3 fill:#e8a838,color:#fff
    style FastAPI fill:#50b86c,color:#fff
    style Chromium fill:#50b86c,color:#fff
    style VNC fill:#50b86c,color:#fff
    style RDS fill:#e8a838,color:#fff
```

## CloudFront Cache Behaviors

| Priority | Path Pattern     | Origin | Caching  | Methods              | Purpose                       |
|----------|-----------------|--------|----------|----------------------|-------------------------------|
| 1        | `/api/v1/vnc/*` | ALB    | Disabled | All + WebSocket      | Persistent VNC WebSocket      |
| 2        | `/api/*`        | ALB    | Disabled | All                  | REST API + event polling      |
| 3        | `/ws/*`         | ALB    | Disabled | All                  | WebSocket connections         |
| 4        | `/health`       | ALB    | Disabled | GET/HEAD/OPTIONS     | Health check                  |
| default  | `/*`            | S3     | 1 hour   | GET/HEAD             | Static frontend (Next.js SPA) |

## API Request Flow (REST)

```mermaid
graph TD
    A["Client: POST /api/v1/tasks/start"] --> B["CloudFront HTTPS termination"]
    B -->|"HTTP :80"| C["ALB, idle timeout 3600s"]
    C -->|"HTTP :8000"| D["EC2 Docker - FastAPI"]
    D --> E["JWT verification via Cognito"]
    E --> F["Create agent task, return task_id"]
    F --> G["Client polls GET /tasks/id/events?since=N\nevery 1.5s"]

    style A fill:#4a90d9,color:#fff
    style B fill:#7b68ee,color:#fff
    style C fill:#7b68ee,color:#fff
    style D fill:#50b86c,color:#fff
    style E fill:#50b86c,color:#fff
    style F fill:#8fbc8f,color:#fff
    style G fill:#4a90d9,color:#fff
```

## VNC Browser Viewer Flow

```mermaid
graph TD
    A["Client opens VNC viewer"] --> B["Frontend constructs WS URL\nwss://domain/api/v1/vnc/ws?task_id=X&token=Y"]
    B --> C["CloudFront routes /api/v1/vnc/*\nCachingDisabled, AllViewer"]
    C --> D["ALB forwards to EC2:8000\nidle timeout 3600s"]
    D --> E["FastAPI VNC proxy\nJWT auth + session lookup"]
    E --> F["websockify\nconnects to x11vnc on localhost"]
    F --> G["Xvfb display streamed\nto client via noVNC"]

    style A fill:#4a90d9,color:#fff
    style B fill:#4a90d9,color:#fff
    style C fill:#7b68ee,color:#fff
    style D fill:#7b68ee,color:#fff
    style E fill:#50b86c,color:#fff
    style F fill:#50b86c,color:#fff
    style G fill:#8fbc8f,color:#fff
```

## Browser Task Execution (Inside Docker)

When a task starts, the following runs inside the Docker container:

```mermaid
graph TD
    A["FastAPI receives start_task"] --> B["AgentService creates CodeAgent"]
    B --> C["CodeAgent launches\nPlaywright + Chromium"]
    C --> D["6-layer kiosk security lockdown"]
    D --> E["Agent executes steps\nbrowse, click, type, extract"]
    E --> F["Events emitted to EventBuffer"]
    F --> G["Client polls /tasks/id/events?since=N"]

    style A fill:#50b86c,color:#fff
    style B fill:#50b86c,color:#fff
    style C fill:#50b86c,color:#fff
    style D fill:#e85d75,color:#fff
    style E fill:#50b86c,color:#fff
    style F fill:#e8a838,color:#fff
    style G fill:#4a90d9,color:#fff
```

Kiosk security layers:
1. Openbox WM (no shortcuts, no decorations, forced maximize)
2. Chromium kiosk mode + enterprise policies
3. X11 key grabber daemon (intercepts Alt+F4, Ctrl+W, etc.)
4. x11vnc input filtering (-input KM, no clipboard)
5. Docker hardening (wget/gnupg removed post-build)
6. Enterprise policies JSON (/etc/chromium/policies/managed/)

## Data Persistence

| Data              | Storage                | Access                         |
|-------------------|------------------------|--------------------------------|
| Chat messages     | PostgreSQL RDS         | Backend via SQLAlchemy async   |
| Conversations     | PostgreSQL RDS         | Backend via SQLAlchemy async   |
| User state        | PostgreSQL RDS         | Backend via SQLAlchemy async   |
| LLM API keys      | SSM Parameter Store    | EC2 fetches at container start |
| Session data      | DynamoDB               | EC2 via IAM instance profile   |

## Authentication Flow

```mermaid
graph TD
    A["User visits CloudFront URL"] --> B["Frontend redirects to\nCognito Hosted UI, PKCE"]
    B --> C["User signs in"]
    C --> D["Cognito redirects to /auth/callback"]
    D --> E["Frontend exchanges code\nfor JWT tokens"]
    E --> F["API requests include JWT\nin Authorization header"]
    F --> G["FastAPI validates JWT\nagainst Cognito User Pool"]

    style A fill:#4a90d9,color:#fff
    style B fill:#4a90d9,color:#fff
    style C fill:#e85d75,color:#fff
    style D fill:#e85d75,color:#fff
    style E fill:#4a90d9,color:#fff
    style F fill:#4a90d9,color:#fff
    style G fill:#50b86c,color:#fff
```

## Why API Gateway Is Bypassed

API Gateway (HTTP API) still exists in Terraform but receives **zero traffic**:

1. **30-second integration timeout**: API Gateway kills long-lived connections (VNC WebSocket, SSE) after 30 seconds. This is a hard limit that cannot be increased.
2. **CloudFront direct-to-ALB**: CloudFront routes all `/api/*` and `/ws/*` traffic directly to the ALB, which has a 3600-second idle timeout.
3. **JWT auth in FastAPI**: Authentication is handled by the backend, not API Gateway.
4. **Candidate for removal**: API Gateway can be removed from Terraform to simplify the architecture.

## EC2 Instance Details

- **Type**: t3.medium (4 GB RAM)
- **Memory**: 4 GB RAM + 2 GB swap = 6 GB addressable
- **Subnet**: Private (no public IP)
- **AMI**: Amazon Linux 2023 (x86_64)
- **Container**: Single Docker container (FastAPI + Playwright + VNC stack)
- **Deployment**: Hot-deploy via SSM RunShellScript (ECR pull + container restart)

## Network Security

| Component    | Inbound Rules                                     |
|-------------|---------------------------------------------------|
| ALB SG      | HTTP :80 from CloudFront prefix list + VPC CIDR   |
| Backend SG  | :8000 from VPC CIDR only                          |
| PostgreSQL  | :5432 from Backend SG only                        |

## Troubleshooting

### Health check fails
```bash
curl -s https://d3p903fxpmjf8v.cloudfront.net/health
# Expected: {"status":"healthy"}
```

### Check EC2 via SSM
```bash
aws ssm send-command --region ca-central-1 \
  --instance-ids i-052e44a607d603f36 \
  --document-name "AWS-RunShellScript" \
  --parameters commands='["docker ps -a","free -m","docker logs openbrowser-backend --tail 50"]'
```

### Common issues
- **504 Gateway Timeout**: EC2 OOM -- check `free -m` via SSM, verify swap exists
- **VNC disconnects**: ALB idle timeout may have been exceeded (3600s limit)
- **SSM command "Delayed"**: EC2 is unresponsive (OOM, disk full) -- reboot via `aws ec2 reboot-instances`
- **Docker ModuleNotFoundError after instance restart**: Stale overlay -- redeploy fresh container from ECR
