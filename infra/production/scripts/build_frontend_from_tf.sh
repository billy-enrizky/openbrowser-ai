#!/bin/bash
set -euo pipefail

# Build frontend using production values from Terraform outputs.
#
# Usage:
#   bash infra/production/scripts/build_frontend_from_tf.sh
#   bash infra/production/scripts/build_frontend_from_tf.sh --deploy
#
# --deploy additionally uploads ./frontend/out to S3 and invalidates CloudFront.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TF_DIR="$PROJECT_ROOT/infra/production/terraform"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
ENV_FILE="$FRONTEND_DIR/.env.production.local"

DEPLOY=false
if [[ "${1:-}" == "--deploy" ]]; then
  DEPLOY=true
fi

if ! command -v terraform >/dev/null 2>&1; then
  echo "ERROR: terraform is not installed."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm is not installed."
  exit 1
fi

echo "--- Reading Terraform outputs ---"
API_URL="$(terraform -chdir="$TF_DIR" output -raw api_base_url)"
WS_URL="$(terraform -chdir="$TF_DIR" output -raw api_ws_url)"
COGNITO_DOMAIN="$(terraform -chdir="$TF_DIR" output -raw cognito_domain_url)"
COGNITO_CLIENT_ID="$(terraform -chdir="$TF_DIR" output -raw cognito_app_client_id)"
FRONTEND_URL="$(terraform -chdir="$TF_DIR" output -raw frontend_url)"

FRONTEND_URL="${FRONTEND_URL%/}"
REDIRECT_URI="$FRONTEND_URL/auth/callback"
LOGOUT_URI="$FRONTEND_URL/login"

echo "--- Writing $ENV_FILE ---"
cat > "$ENV_FILE" <<EOF
NEXT_PUBLIC_API_URL=$API_URL
NEXT_PUBLIC_WS_URL=$WS_URL
NEXT_PUBLIC_AUTH_ENABLED=true
NEXT_PUBLIC_COGNITO_DOMAIN=$COGNITO_DOMAIN
NEXT_PUBLIC_COGNITO_CLIENT_ID=$COGNITO_CLIENT_ID
NEXT_PUBLIC_COGNITO_REDIRECT_URI=$REDIRECT_URI
NEXT_PUBLIC_COGNITO_LOGOUT_URI=$LOGOUT_URI
NEXT_PUBLIC_COGNITO_SCOPES=openid email profile
EOF

echo "--- Building frontend ---"
cd "$FRONTEND_DIR"
npm ci
npm run build

if [[ "$DEPLOY" == "true" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "ERROR: aws CLI is required for --deploy."
    exit 1
  fi

  BUCKET="$(terraform -chdir="$TF_DIR" output -raw frontend_s3_bucket)"
  DIST_ID="$(terraform -chdir="$TF_DIR" output -raw cloudfront_distribution_id)"

  echo "--- Deploying static assets to S3 ---"
  aws s3 sync "$FRONTEND_DIR/out/" "s3://$BUCKET/" --delete

  echo "--- Invalidating CloudFront cache ---"
  aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" >/dev/null
fi

echo "=== Frontend build complete ==="
echo "Environment file: $ENV_FILE"
