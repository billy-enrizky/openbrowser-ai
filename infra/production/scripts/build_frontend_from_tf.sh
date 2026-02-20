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

tf_output_raw() {
  local key="$1"
  terraform -chdir="$TF_DIR" output -raw "$key"
}

echo "--- Reading Terraform outputs ---"
CF_DOMAIN="$(tf_output_raw frontend_url | sed 's|https://||')"
API_URL="https://${CF_DOMAIN}"
WS_URL="wss://${CF_DOMAIN}"
COGNITO_DOMAIN="$(tf_output_raw cognito_domain_url)"
COGNITO_CLIENT_ID="$(tf_output_raw cognito_app_client_id)"
REDIRECT_URI="$(tf_output_raw cognito_primary_callback_url)"
LOGOUT_URI="$(tf_output_raw cognito_primary_logout_url)"
COGNITO_SCOPES="$(tf_output_raw cognito_oauth_scopes_string)"

echo "--- Writing $ENV_FILE ---"
cat > "$ENV_FILE" <<EOF
NEXT_PUBLIC_API_URL=$API_URL
NEXT_PUBLIC_WS_URL=$WS_URL
NEXT_PUBLIC_AUTH_ENABLED=true
NEXT_PUBLIC_COGNITO_DOMAIN=$COGNITO_DOMAIN
NEXT_PUBLIC_COGNITO_CLIENT_ID=$COGNITO_CLIENT_ID
NEXT_PUBLIC_COGNITO_REDIRECT_URI=$REDIRECT_URI
NEXT_PUBLIC_COGNITO_LOGOUT_URI=$LOGOUT_URI
NEXT_PUBLIC_COGNITO_SCOPES=$COGNITO_SCOPES
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

  BUCKET="$(tf_output_raw frontend_s3_bucket)"
  DIST_ID="$(tf_output_raw cloudfront_distribution_id)"
  REGION="$(tf_output_raw aws_region)"

  # Upload hashed static assets with immutable 1-year cache.
  # _next/static/* filenames contain content hashes; they change on every
  # build so caching forever is safe and avoids redundant downloads.
  echo "--- Uploading hashed static assets (immutable, 1-year cache) ---"
  aws s3 sync "$FRONTEND_DIR/out/_next/static/" "s3://$BUCKET/_next/static/" \
    --cache-control "public, max-age=31536000, immutable" \
    --delete \
    --region "$REGION"

  # Upload HTML and non-hashed files with no-cache.
  # HTML references hashed JS/CSS bundles by name. If HTML is cached,
  # users may load stale HTML pointing to deleted bundles -- this
  # prevents that by forcing browsers to always revalidate.
  echo "--- Uploading HTML and non-hashed files (no-cache) ---"
  aws s3 sync "$FRONTEND_DIR/out/" "s3://$BUCKET/" \
    --cache-control "no-cache, no-store, must-revalidate" \
    --exclude "_next/static/*" \
    --delete \
    --region "$REGION"

  echo "--- Invalidating CloudFront cache ---"
  aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" >/dev/null
fi

echo "=== Frontend build complete ==="
echo "Environment file: $ENV_FILE"
