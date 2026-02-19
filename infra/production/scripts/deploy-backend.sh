#!/usr/bin/env bash
# Deploy backend Docker image to ECR and restart on EC2.
#
# Usage: ./infra/production/scripts/deploy-backend.sh
#
# Loads API keys from .env in the project root.
# Requires: AWS CLI, Docker, and valid AWS credentials.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Infrastructure constants ───────────────────────────────────────
ECR_REPO="529206289231.dkr.ecr.ca-central-1.amazonaws.com/openbrowser-backend"
EC2_INSTANCE="i-052e44a607d603f36"
AWS_REGION="ca-central-1"

# RDS
DB_HOST="openbrowser-postgres.cvqwa2aoyljt.ca-central-1.rds.amazonaws.com"
DB_PORT="5432"
DB_NAME="openbrowser"
DB_USER="openbrowser"
DB_PASS="REDACTED_DB_PASSWORD"
DATABASE_URL="postgresql+asyncpg://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Cognito
COGNITO_USER_POOL_ID="ca-central-1_uU8gXJMFW"
COGNITO_CLIENT_ID="5ovv2tf4r12f6m0q3kbh83nj6f"
COGNITO_REGION="ca-central-1"

# ── Load API keys from .env ────────────────────────────────────────
ENV_FILE="$PROJECT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env file not found at $ENV_FILE"
  exit 1
fi

# Source only the keys we need (handles both single and double quotes)
extract_env() {
  local key="$1"
  local val
  val=$(grep -E "^${key}=" "$ENV_FILE" | tail -1 | sed "s/^${key}=//" | sed "s/^['\"]//;s/['\"]$//")
  if [[ -z "$val" ]]; then
    echo "ERROR: $key not found in .env"
    exit 1
  fi
  echo "$val"
}

GOOGLE_API_KEY=$(extract_env "GOOGLE_API_KEY")
OPENAI_API_KEY=$(extract_env "OPENAI_API_KEY")

echo "Loaded API keys from .env"

cd "$PROJECT_DIR"

# ── 1. Build Docker image ──────────────────────────────────────────
echo ""
echo "[1/5] Building Docker image (linux/amd64)..."
docker build --platform linux/amd64 -t openbrowser-backend -f backend/Dockerfile .

# ── 2. Login to ECR ────────────────────────────────────────────────
echo ""
echo "[2/5] Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REPO"

# ── 3. Tag and push ────────────────────────────────────────────────
echo ""
echo "[3/5] Pushing image to ECR..."
docker tag openbrowser-backend:latest "$ECR_REPO:latest"
docker push "$ECR_REPO:latest"

# ── 4. Deploy on EC2 via SSM ───────────────────────────────────────
echo ""
echo "[4/5] Deploying on EC2 ($EC2_INSTANCE)..."

# Build the docker run command with all env vars.
# Uses a heredoc so quotes and special chars are handled correctly.
DEPLOY_COMMANDS=$(cat <<CMDS
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO
docker pull $ECR_REPO:latest
docker stop openbrowser-backend || true
docker rm openbrowser-backend || true
docker run -d \
  --name openbrowser-backend \
  --restart unless-stopped \
  -p 8000:8000 \
  --shm-size=2g \
  -e DATABASE_URL="$DATABASE_URL" \
  -e COGNITO_USER_POOL_ID=$COGNITO_USER_POOL_ID \
  -e COGNITO_CLIENT_ID=$COGNITO_CLIENT_ID \
  -e COGNITO_REGION=$COGNITO_REGION \
  -e AWS_DEFAULT_REGION=$AWS_REGION \
  -e GOOGLE_API_KEY=$GOOGLE_API_KEY \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  $ECR_REPO:latest
sleep 8
curl -sf http://localhost:8000/health || echo "HEALTH CHECK FAILED"
CMDS
)

COMMAND_ID=$(aws ssm send-command \
  --instance-ids "$EC2_INSTANCE" \
  --document-name "AWS-RunShellScript" \
  --parameters "commands=[$(echo "$DEPLOY_COMMANDS" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().strip().split("\n")))[1:-1]')]" \
  --region "$AWS_REGION" \
  --query 'Command.CommandId' \
  --output text)

echo "  SSM Command: $COMMAND_ID"
echo "  Waiting for deployment..."

# Poll until command completes
for i in $(seq 1 30); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$COMMAND_ID" \
    --instance-id "$EC2_INSTANCE" \
    --region "$AWS_REGION" \
    --query 'Status' \
    --output text 2>/dev/null || echo "Pending")

  if [[ "$STATUS" == "Success" ]]; then
    break
  elif [[ "$STATUS" == "Failed" || "$STATUS" == "Cancelled" || "$STATUS" == "TimedOut" ]]; then
    echo "ERROR: SSM command $STATUS"
    aws ssm get-command-invocation \
      --command-id "$COMMAND_ID" \
      --instance-id "$EC2_INSTANCE" \
      --region "$AWS_REGION" \
      --query 'StandardErrorContent' \
      --output text
    exit 1
  fi

  sleep 5
done

# ── 5. Verify ──────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying deployment..."

OUTPUT=$(aws ssm get-command-invocation \
  --command-id "$COMMAND_ID" \
  --instance-id "$EC2_INSTANCE" \
  --region "$AWS_REGION" \
  --query 'StandardOutputContent' \
  --output text)

if echo "$OUTPUT" | grep -q '"status":"healthy"'; then
  echo ""
  echo "Deployment successful. Backend is healthy."
else
  echo ""
  echo "WARNING: Health check may have failed. SSM output:"
  echo "$OUTPUT"
  exit 1
fi
