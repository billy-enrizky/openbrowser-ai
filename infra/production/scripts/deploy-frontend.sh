#!/usr/bin/env bash
set -euxo pipefail

# Frontend deployment entrypoint for OpenBrowser-AI production.
# Delegates to the Terraform-output-driven build/deploy script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

bash "$SCRIPT_DIR/build_frontend_from_tf.sh" --deploy
