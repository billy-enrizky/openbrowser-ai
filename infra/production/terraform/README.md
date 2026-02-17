# OpenBrowser-AI Production Infrastructure (Terraform)

This directory defines AWS infrastructure for OpenBrowser-AI: VPC, single EC2 backend (Docker), internal ALB, API Gateway (HTTP API + VPC Link), Cognito, DynamoDB, ECR, Secrets Manager, and S3 + CloudFront for the frontend.

## Architecture

```
Internet
   │
   ▼
API Gateway (HTTP API, optional JWT)
   │  VPC Link
   ▼
Internal ALB (:80)
   │
   ▼
EC2 (private subnet) — Docker backend :8000
   │
   ├── DynamoDB (sessions)
   ├── Secrets Manager (LLM keys)
   └── ECR (pull image)
```

- **Frontend**: Static Next.js export on S3, served via CloudFront.
- **Backend**: One EC2 instance in a private subnet; user_data pulls the image from ECR and runs the container with env from Secrets Manager and DynamoDB table name.

## File layout

| File | Purpose |
|------|--------|
| `versions.tf` | Terraform/provider requirements, AWS provider, data sources |
| `vpc.tf` | VPC, public/private subnets, NAT, route tables |
| `security_groups.tf` | ALB and backend EC2 security groups |
| `alb.tf` | Internal ALB, target group, HTTP listener |
| `backend.tf` | EC2 instance, user_data, target group attachment |
| `api_gateway.tf` | HTTP API, VPC Link, routes, optional JWT authorizer |
| `iam.tf` | Backend EC2 IAM role, ECR/DynamoDB/Secrets/SSM permissions |
| `ecr.tf` | ECR repository and lifecycle policy for backend image |
| `dynamodb.tf` | Sessions table + VPC endpoint |
| `secrets.tf` | Optional Secrets Manager secret for LLM keys |
| `cognito.tf` | User pool, app client, hosted domain |
| `frontend.tf` | S3 bucket, CloudFront distribution, OAC |
| `main.tf` | SSM parameters for API keys (Google, OpenAI, Anthropic) |
| `outputs.tf` | URLs and IDs (API, frontend, Cognito, ECR, etc.) |
| `variables.tf` | Input variables |
| `AUTHENTICATION.md` | API Gateway + Cognito JWT setup and frontend integration |
| `COGNITO_SETUP.md` | When callback URLs are needed for Cognito |

## Prerequisites

- [Terraform](https://www.terraform.io/downloads) >= 1.5
- AWS CLI configured (e.g. `aws configure`), with credentials for the target account/region
- Docker (for building and pushing the backend image to ECR)

## Quick start

1. **Copy and edit variables**

   ```bash
   cd infra/production/terraform
   cp terraform.tfvars.example terraform.tfvars
   # Edit terraform.tfvars (aws_region, project_name, backend_image = "" to use ECR).
   ```

2. **Create ECR and push backend image (recommended order)**

   ```bash
   terraform init
   terraform apply -target=aws_ecr_repository.backend
   REPO=$(terraform output -raw backend_ecr_repository_url)
   REGION=ca-central-1   # or your aws_region
   aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${REPO%%/*}
   # From repo root:
   docker build -t openbrowser-backend -f backend/Dockerfile .
   docker tag openbrowser-backend:latest $REPO:latest
   docker push $REPO:latest
   ```

3. **Apply full infrastructure**

   ```bash
   terraform plan
   terraform apply
   ```

4. **Set LLM API keys (Secrets Manager or SSM)**

   - If using the Terraform-created secret: put keys in Secrets Manager secret (see `backend_secret_name` output).
   - SSM parameters are in `main.tf`; set values via AWS Console or CLI (see outputs `ssm_*_api_key_name`).

5. **Verify**

   ```bash
   curl -i "$(terraform output -raw api_base_url)health"
   ```

## What’s included

| Component | Purpose |
|-----------|--------|
| **VPC** | Public/private subnets (2 AZs), NAT, DynamoDB VPC endpoint |
| **EC2 backend** | Single instance in private subnet; Docker runs backend image from ECR |
| **ALB** | Internal ALB, health check `/health`, forwards to backend:8000 |
| **API Gateway** | HTTP API with VPC Link to ALB; public `GET /health`, optional JWT on other routes |
| **Cognito** | User pool, app client, hosted domain (see `AUTHENTICATION.md` / `COGNITO_SETUP.md`) |
| **DynamoDB** | Table `{project_name}-sessions` (pk/sk); `DDB_TABLE` passed to container |
| **ECR** | Backend image repo; EC2 pulls from here when `backend_image` is empty |
| **Secrets Manager** | Optional secret for LLM keys; EC2 injects into container env |
| **S3 + CloudFront** | Static frontend; CloudFront OAC, SPA 403/404 → index.html |

## Variables

See `variables.tf`. Key ones:

- **backend_image** — Leave `""` to use the ECR repo Terraform creates; set to a full URI to use another registry.
- **backend_image_tag** — Tag to pull from the Terraform ECR repo (default `latest`).
- **backend_port** — Port the backend listens on (default `8000`).
- **enable_backend_auth** — Set `true` to require Cognito JWT in FastAPI (REST + WebSocket).
- **enable_api_auth** — Optional second JWT check at API Gateway (kept `false` by default to simplify WebSocket auth).
- **secrets_manager_secret_name** — Leave empty to create a placeholder secret, or set existing secret name/ARN.
- **frontend_domain_name** / **frontend_acm_certificate_arn** — Optional custom domain (ACM in us-east-1 for CloudFront).
- **cognito_callback_urls** / **cognito_logout_urls** — Optional explicit OAuth URLs. If empty, Terraform auto-generates URLs from frontend domain + localhost.

## Backend image (ECR-first workflow)

1. Create only the ECR repo:  
   `terraform apply -target=aws_ecr_repository.backend`
2. Log in to ECR, build from **repo root**, tag and push (see Quick start above).
3. Keep `backend_image = ""` and `backend_image_tag = "latest"` (or your tag) in `terraform.tfvars`.
4. Run full `terraform apply`; EC2 user_data will pull `backend_ecr_repository_url:backend_image_tag` on boot.

To use a different registry, set `backend_image` to the full image URI in `terraform.tfvars`.

## Auth (Cognito + Hosted UI + PKCE)

- **AUTHENTICATION.md** — Exact env values and deploy steps for this repository's frontend PKCE implementation.
- **COGNITO_SETUP.md** — Callback/logout URL behavior and when to override Terraform defaults.

For parity with local auth, use:
- `enable_backend_auth = true`
- `enable_api_auth = false`

## Frontend deploy

From project root, after `terraform apply`:

```bash
export NEXT_PUBLIC_API_URL=$(terraform -chdir=infra/production/terraform output -raw api_base_url)
export NEXT_PUBLIC_WS_URL=$(terraform -chdir=infra/production/terraform output -raw api_ws_url)
export NEXT_PUBLIC_AUTH_ENABLED=true
export NEXT_PUBLIC_COGNITO_DOMAIN=$(terraform -chdir=infra/production/terraform output -raw cognito_domain_url)
export NEXT_PUBLIC_COGNITO_CLIENT_ID=$(terraform -chdir=infra/production/terraform output -raw cognito_app_client_id)
export NEXT_PUBLIC_COGNITO_REDIRECT_URI=https://app.example.com/auth/callback/
export NEXT_PUBLIC_COGNITO_LOGOUT_URI=https://app.example.com/login/
export NEXT_PUBLIC_COGNITO_SCOPES="openid email profile"
cd frontend && npm ci && npm run build
aws s3 sync out/ s3://$(terraform -chdir=infra/production/terraform output -raw frontend_s3_bucket)/ --delete
aws cloudfront create-invalidation --distribution-id $(terraform -chdir=infra/production/terraform output -raw cloudfront_distribution_id) --paths "/*"
```

## Outputs

- `api_base_url` / `api_ws_url` — Use for `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_WS_URL`.
- `frontend_url`, `frontend_s3_bucket`, `cloudfront_distribution_id` — Frontend hosting.
- `backend_ecr_repository_url`, `backend_image_uri` — Where to push the image and what EC2 pulls.
- `cognito_user_pool_id`, `cognito_app_client_id`, `cognito_domain_url`, `cognito_callback_urls`, `cognito_logout_urls` — Auth integration.
- `dynamodb_table_name`, `backend_secret_name` — Backend config.

## Troubleshooting

- **Health check fails** — Ensure backend serves `GET /health` with 200; check ALB target group health in the console.
- **API Gateway 403/502** — Verify VPC Link status, security groups (ALB → backend port), and that the backend is healthy.
- **EC2 not pulling image** — Confirm image exists in ECR (`aws ecr describe-images --repository-name openbrowser-backend --region ca-central-1`) and IAM role has ECR read.

## Cleanup

```bash
terraform destroy
```

Removes all resources created by this configuration.
