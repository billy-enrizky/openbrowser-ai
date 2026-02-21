#!/bin/bash
set -euo pipefail

# Deploy the landing page to S3 + CloudFront.
#
# Usage:
#   bash infra/production/scripts/deploy-landing.sh
#
# Reads infrastructure values from Terraform outputs (zero hardcoded values).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TF_DIR="$PROJECT_ROOT/infra/production/terraform"
LANDING_DIR="$PROJECT_ROOT/landing"

if ! command -v terraform >/dev/null 2>&1; then
  echo "ERROR: terraform is not installed."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm is not installed."
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI is not installed."
  exit 1
fi

tf_output_raw() {
  terraform -chdir="$TF_DIR" output -raw "$1"
}

echo "--- Reading Terraform outputs ---"
WAITLIST_API_URL="$(tf_output_raw waitlist_api_url)"
BUCKET="$(tf_output_raw landing_s3_bucket)"
DIST_ID="$(tf_output_raw landing_cloudfront_distribution_id)"
REGION="$(tf_output_raw aws_region)"

echo "  Waitlist API: $WAITLIST_API_URL"
echo "  S3 Bucket:    $BUCKET"
echo "  CloudFront:   $DIST_ID"
echo "  Region:       $REGION"

echo "--- Writing .env.production.local ---"
cat > "$LANDING_DIR/.env.production.local" <<EOF
NEXT_PUBLIC_WAITLIST_API_URL=$WAITLIST_API_URL
EOF

echo "--- Building landing page ---"
cd "$LANDING_DIR"
npm ci
npm run build

echo "--- Uploading hashed static assets (immutable, 1-year cache) ---"
aws s3 sync "$LANDING_DIR/out/_next/static/" "s3://$BUCKET/_next/static/" \
  --cache-control "public, max-age=31536000, immutable" \
  --delete \
  --region "$REGION"

echo "--- Uploading HTML and non-hashed files (no-cache) ---"
aws s3 sync "$LANDING_DIR/out/" "s3://$BUCKET/" \
  --cache-control "no-cache, no-store, must-revalidate" \
  --exclude "_next/static/*" \
  --delete \
  --region "$REGION"

echo "--- Invalidating CloudFront cache ---"
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" >/dev/null

echo "=== Landing page deployed ==="
echo "URL: $(tf_output_raw landing_url)"
