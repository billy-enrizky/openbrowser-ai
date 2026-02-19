#!/usr/bin/env bash
# Deploy frontend to S3 with proper cache headers and CloudFront invalidation.
#
# Usage: ./infra/production/scripts/deploy-frontend.sh
#
# Requires: AWS CLI configured with appropriate credentials.

set -euo pipefail

BUCKET="openbrowser-frontend-529206289231"
DISTRIBUTION_ID="EZ13W3YNGIZJM"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$PROJECT_DIR/frontend/out"

cd "$PROJECT_DIR"

# ── 1. Build frontend ──────────────────────────────────────────────
echo "[1/4] Building frontend..."
(cd frontend && npm run build)

# ── 2. Sync hashed static assets (long cache) ─────────────────────
# _next/static/* files contain content hashes in their filenames.
# They can be cached indefinitely -- a new build produces new filenames.
echo "[2/4] Uploading hashed static assets (immutable, 1 year cache)..."
aws s3 sync "$BUILD_DIR/_next/static/" "s3://$BUCKET/_next/static/" \
  --cache-control "public, max-age=31536000, immutable" \
  --delete \
  --region ca-central-1

# ── 3. Sync HTML and other non-hashed files (no cache) ─────────────
# HTML files reference hashed JS/CSS. They must not be cached so
# browsers always fetch the latest version pointing to new bundles.
echo "[3/4] Uploading HTML and non-hashed files (no-cache)..."
aws s3 sync "$BUILD_DIR/" "s3://$BUCKET/" \
  --cache-control "no-cache, no-store, must-revalidate" \
  --exclude "_next/static/*" \
  --delete \
  --region ca-central-1

# ── 4. Invalidate CloudFront cache ─────────────────────────────────
# Invalidate HTML paths. Hashed assets don't need invalidation because
# their filenames change, but we invalidate root paths to be safe.
echo "[4/4] Invalidating CloudFront cache..."
INVALIDATION_ID=$(aws cloudfront create-invalidation \
  --distribution-id "$DISTRIBUTION_ID" \
  --paths "/*" \
  --region us-east-1 \
  --query 'Invalidation.Id' \
  --output text)

echo ""
echo "Deployment complete."
echo "  CloudFront invalidation: $INVALIDATION_ID"
echo "  URL: https://d3p903fxpmjf8v.cloudfront.net"
