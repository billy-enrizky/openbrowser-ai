#!/usr/bin/env bash
set -euxo pipefail

# Frontend deployment script for OpenBrowser-AI production
# - Reads API + frontend outputs from Terraform
# - Builds Next.js frontend with correct env vars
# - Syncs static export to S3
# - Invalidates CloudFront cache

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="$ROOT_DIR/infra/production/terraform"
REGION="ca-central-1"

cd "$TF_DIR"

# 1) Read outputs from Terraform
API_BASE_URL="$(terraform output -raw api_base_url)"
API_WS_URL="$(terraform output -raw api_ws_url)"
S3_BUCKET="$(terraform output -raw frontend_s3_bucket)"
CLOUDFRONT_DIST_ID="$(terraform output -raw cloudfront_distribution_id)"

# 2) Build frontend with correct env vars
cd "$ROOT_DIR/frontend"
export NEXT_PUBLIC_API_URL="$API_BASE_URL"
export NEXT_PUBLIC_WS_URL="$API_WS_URL"

npm ci
npm run build   # outputs to ./out/

# 3) Sync build output to S3
cd "$ROOT_DIR"
aws s3 sync "frontend/out/" "s3://$S3_BUCKET/" --delete --region "$REGION"

# 4) Invalidate CloudFront cache so changes are visible immediately
aws cloudfront create-invalidation \
  --distribution-id "$CLOUDFRONT_DIST_ID" \
  --paths "/*" \
  --region "$REGION"

