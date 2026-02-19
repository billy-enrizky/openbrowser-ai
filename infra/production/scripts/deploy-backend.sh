#!/usr/bin/env bash
set -euxo pipefail

# Backend deployment script for OpenBrowser production.
#
# All infrastructure values are read from Terraform outputs (IaC).
# API keys are read from SSM Parameter Store (pushed via push_api_keys.sh).
#
# Steps:
#   1. Build Docker image (linux/amd64)
#   2. Push to ECR
#   3. Hot-deploy on EC2 via SSM RunShellScript (pull + restart container)
#   4. Verify health check
#
# Usage:
#   bash infra/production/scripts/deploy-backend.sh
#
# Prerequisites:
#   - terraform, docker, aws CLI
#   - Valid AWS credentials
#   - API keys pushed to SSM (run push_api_keys.sh first if needed)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TF_DIR="$PROJECT_ROOT/infra/production/terraform"

# ── Helpers ────────────────────────────────────────────────────────
tf_output_raw() {
  terraform -chdir="$TF_DIR" output -raw "$1"
}

# ── Read all values from Terraform ─────────────────────────────────
echo "=== Reading Terraform outputs ==="
REGION="$(tf_output_raw aws_region)"
ECR_REPO="$(tf_output_raw backend_ecr_repository_url)"
IMAGE_URI="$(tf_output_raw backend_image_uri)"
INSTANCE_ID="$(tf_output_raw backend_instance_id)"
BACKEND_PORT=8000

# Database
DB_HOST="$(tf_output_raw postgres_endpoint)"
DB_PORT="$(tf_output_raw postgres_port)"
DB_NAME="$(tf_output_raw postgres_db_name)"
DB_USER="$(tf_output_raw postgres_username)"
DB_PASS="$(terraform -chdir="$TF_DIR" output -raw postgres_password)"

DATABASE_URL="postgresql+asyncpg://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Cognito
COGNITO_POOL="$(tf_output_raw cognito_user_pool_id)"
COGNITO_CLIENT="$(tf_output_raw cognito_app_client_id)"

# SSM parameter names for API keys
SSM_GOOGLE="$(tf_output_raw ssm_google_api_key_name)"
SSM_OPENAI="$(tf_output_raw ssm_openai_api_key_name)"
SSM_ANTHROPIC="$(tf_output_raw ssm_anthropic_api_key_name)"

echo "  ECR:       $ECR_REPO"
echo "  Image:     $IMAGE_URI"
echo "  Instance:  $INSTANCE_ID"
echo "  Region:    $REGION"

# ── 1. Build Docker image ──────────────────────────────────────────
echo ""
echo "=== [1/4] Building Docker image (linux/amd64) ==="
cd "$PROJECT_ROOT"
docker build --platform linux/amd64 -t openbrowser-backend -f backend/Dockerfile .

# ── 2. Push to ECR ─────────────────────────────────────────────────
echo ""
echo "=== [2/4] Pushing to ECR ==="
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${ECR_REPO%%/*}"

docker tag openbrowser-backend:latest "$IMAGE_URI"
docker push "$IMAGE_URI"

# ── 3. Hot-deploy on EC2 via SSM ───────────────────────────────────
echo ""
echo "=== [3/4] Deploying on EC2 ($INSTANCE_ID) ==="

# The EC2 instance fetches API keys directly from SSM Parameter Store
# at container startup, matching the pattern in backend-userdata.sh.
read -r -d '' DEPLOY_SCRIPT <<REMOTEOF || true
#!/bin/bash
set -euxo pipefail

# Login to ECR
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin ${ECR_REPO%%/*}

# Pull latest image
docker pull $IMAGE_URI

# Fetch API keys from SSM Parameter Store
GOOGLE_API_KEY=\$(aws ssm get-parameter --name "$SSM_GOOGLE" --with-decryption --region "$REGION" --query "Parameter.Value" --output text 2>/dev/null || echo "")
OPENAI_API_KEY=\$(aws ssm get-parameter --name "$SSM_OPENAI" --with-decryption --region "$REGION" --query "Parameter.Value" --output text 2>/dev/null || echo "")
ANTHROPIC_API_KEY=\$(aws ssm get-parameter --name "$SSM_ANTHROPIC" --with-decryption --region "$REGION" --query "Parameter.Value" --output text 2>/dev/null || echo "")

# Stop and remove old container
docker stop openbrowser-backend || true
docker rm openbrowser-backend || true

# Start new container with all env vars
docker run -d \
  --name openbrowser-backend \
  --restart unless-stopped \
  -p $BACKEND_PORT:$BACKEND_PORT \
  --shm-size=2g \
  --security-opt seccomp=unconfined \
  -e DATABASE_URL="$DATABASE_URL" \
  -e COGNITO_USER_POOL_ID=$COGNITO_POOL \
  -e COGNITO_APP_CLIENT_ID=$COGNITO_CLIENT \
  -e COGNITO_REGION=$REGION \
  -e AWS_DEFAULT_REGION=$REGION \
  -e AUTH_ENABLED=true \
  -e VNC_ENABLED=true \
  -e VNC_WIDTH=1920 \
  -e VNC_HEIGHT=1080 \
  -e GOOGLE_API_KEY=\$GOOGLE_API_KEY \
  -e OPENAI_API_KEY=\$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=\$ANTHROPIC_API_KEY \
  $IMAGE_URI

# Wait for startup
sleep 8
curl -sf http://localhost:$BACKEND_PORT/health || echo "HEALTH CHECK FAILED"
REMOTEOF

# Send the deploy script to EC2 via SSM
COMMAND_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "commands=[$(echo "$DEPLOY_SCRIPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip().split("\n")))[1:-1]')]" \
  --timeout-seconds 300 \
  --region "$REGION" \
  --query 'Command.CommandId' \
  --output text)

echo "  SSM Command: $COMMAND_ID"
echo "  Waiting for deployment to complete..."

# Poll until done
for i in $(seq 1 40); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' \
    --output text 2>/dev/null || echo "Pending")

  case "$STATUS" in
    Success) break ;;
    Failed|Cancelled|TimedOut)
      echo "ERROR: SSM command $STATUS"
      aws ssm get-command-invocation \
        --command-id "$COMMAND_ID" \
        --instance-id "$INSTANCE_ID" \
        --region "$REGION" \
        --query 'StandardErrorContent' \
        --output text
      exit 1
      ;;
  esac
  sleep 5
done

# ── 4. Verify ──────────────────────────────────────────────────────
echo ""
echo "=== [4/4] Verifying deployment ==="

OUTPUT=$(aws ssm get-command-invocation \
  --command-id "$COMMAND_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardOutputContent' \
  --output text)

if echo "$OUTPUT" | grep -q '"status":"healthy"'; then
  echo ""
  echo "Deployment successful. Backend is healthy."
  echo "  Instance: $INSTANCE_ID"
  echo "  Image:    $IMAGE_URI"
else
  echo ""
  echo "WARNING: Health check may have failed. SSM output:"
  echo "$OUTPUT"
  exit 1
fi
