#!/bin/bash
set -euo pipefail

# Push API keys from local .env to SSM Parameter Store.
#
# SSM parameter names and region are read from Terraform outputs.
#
# Usage:
#   bash infra/production/scripts/push_api_keys.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TF_DIR="$PROJECT_ROOT/infra/production/terraform"
ENV_FILE="$PROJECT_ROOT/.env"

if ! command -v terraform >/dev/null 2>&1; then
    echo "ERROR: terraform is not installed."
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

tf_output_raw() {
    terraform -chdir="$TF_DIR" output -raw "$1"
}

echo "--- Reading Terraform outputs ---"
AWS_REGION="$(tf_output_raw aws_region)"
SSM_GOOGLE="$(tf_output_raw ssm_google_api_key_name)"
SSM_OPENAI="$(tf_output_raw ssm_openai_api_key_name)"
SSM_ANTHROPIC="$(tf_output_raw ssm_anthropic_api_key_name)"

echo "--- Syncing API keys from .env to SSM Parameter Store ---"

# Extract a key value from the .env file, stripping surrounding quotes.
extract_env_value() {
    local key_name="$1"
    local value
    value=$(grep "^${key_name}=" "$ENV_FILE" | head -1 | cut -d= -f2-)
    # Strip surrounding single or double quotes
    value="${value#\"}"
    value="${value%\"}"
    value="${value#\'}"
    value="${value%\'}"
    echo "$value"
}

push_key_to_ssm() {
    local env_key="$1"
    local ssm_name="$2"
    local value
    value="$(extract_env_value "$env_key")"

    if [ -z "$value" ] || [ "$value" = "PLACEHOLDER" ] || [ "$value" = "replace-me" ]; then
        echo "  SKIP: $env_key not set in .env"
        return
    fi

    aws ssm put-parameter \
        --name "$ssm_name" \
        --value "$value" \
        --type SecureString \
        --overwrite \
        --region "$AWS_REGION" \
        --output text > /dev/null 2>&1

    echo "  OK: $env_key -> $ssm_name"
}

push_key_to_ssm "GOOGLE_API_KEY"    "$SSM_GOOGLE"
push_key_to_ssm "OPENAI_API_KEY"    "$SSM_OPENAI"
push_key_to_ssm "ANTHROPIC_API_KEY" "$SSM_ANTHROPIC"

# Fallback: push GEMINI_API_KEY as GOOGLE_API_KEY if GOOGLE is not set
GEMINI_VALUE="$(extract_env_value "GEMINI_API_KEY")"
GOOGLE_VALUE="$(extract_env_value "GOOGLE_API_KEY")"
if [ -n "$GEMINI_VALUE" ] && [ "$GEMINI_VALUE" != "PLACEHOLDER" ] && [ -z "$GOOGLE_VALUE" ]; then
    aws ssm put-parameter \
        --name "$SSM_GOOGLE" \
        --value "$GEMINI_VALUE" \
        --type SecureString \
        --overwrite \
        --region "$AWS_REGION" \
        --output text > /dev/null 2>&1
    echo "  OK: GEMINI_API_KEY -> $SSM_GOOGLE"
fi

echo ""
echo "=== API keys synced to SSM ($AWS_REGION) ==="
