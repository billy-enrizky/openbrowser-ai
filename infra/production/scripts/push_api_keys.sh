#!/bin/bash
set -euo pipefail

# Push API keys from local .env to SSM Parameter Store
#
# Usage:
#   bash infra/production/scripts/push_api_keys.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
AWS_REGION="${AWS_REGION:-ca-central-1}"
PROJECT_NAME="${PROJECT_NAME:-openbrowser}"

echo "--- Syncing API keys from .env to SSM Parameter Store ---"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

push_key_to_ssm() {
    local key_name="$1"
    local ssm_name="$2"
    local value
    value=$(grep "^${key_name}=" "$ENV_FILE" | head -1 | cut -d= -f2-)

    if [ -z "$value" ] || [ "$value" = "PLACEHOLDER" ]; then
        echo "  SKIP: $key_name not set in .env"
        return
    fi

    aws ssm put-parameter \
        --name "$ssm_name" \
        --value "$value" \
        --type SecureString \
        --overwrite \
        --region "$AWS_REGION" \
        --output text > /dev/null 2>&1

    echo "  OK: $key_name -> $ssm_name"
}

push_key_to_ssm "GOOGLE_API_KEY"    "/${PROJECT_NAME}/GOOGLE_API_KEY"
push_key_to_ssm "OPENAI_API_KEY"    "/${PROJECT_NAME}/OPENAI_API_KEY"
push_key_to_ssm "ANTHROPIC_API_KEY" "/${PROJECT_NAME}/ANTHROPIC_API_KEY"

# Also push GEMINI_API_KEY if present (some configs use this name)
GEMINI_KEY=$(grep "^GEMINI_API_KEY=" "$ENV_FILE" | head -1 | cut -d= -f2- || echo "")
if [ -n "$GEMINI_KEY" ] && [ "$GEMINI_KEY" != "PLACEHOLDER" ]; then
    GOOGLE_KEY=$(grep "^GOOGLE_API_KEY=" "$ENV_FILE" | head -1 | cut -d= -f2- || echo "")
    if [ -z "$GOOGLE_KEY" ]; then
        aws ssm put-parameter \
            --name "/${PROJECT_NAME}/GOOGLE_API_KEY" \
            --value "$GEMINI_KEY" \
            --type SecureString \
            --overwrite \
            --region "$AWS_REGION" \
            --output text > /dev/null 2>&1
        echo "  OK: GEMINI_API_KEY -> /${PROJECT_NAME}/GOOGLE_API_KEY"
    fi
fi

echo ""
echo "=== API keys synced successfully ==="
