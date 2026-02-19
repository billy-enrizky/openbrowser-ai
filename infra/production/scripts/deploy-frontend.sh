#!/usr/bin/env bash
set -euxo pipefail

# Frontend deployment entrypoint for OpenBrowser production.
#
# Reads Terraform outputs for all infrastructure values, builds the
# frontend with correct env vars, uploads to S3 with proper cache
# headers, and invalidates CloudFront.
#
# Usage:
#   bash infra/production/scripts/deploy-frontend.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/build_frontend_from_tf.sh" --deploy
