#!/bin/bash
set -euo pipefail

# Launch an eval spot instance from the Terraform launch template.
#
# Usage:
#   bash infra/eval/scripts/launch_eval.sh
#
# Prerequisites:
#   - terraform apply has been run (infra/eval/terraform/)
#   - AWS CLI configured with credentials
#   - .env file with API keys at project root

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TF_DIR="$SCRIPT_DIR/../terraform"
ENV_FILE="$PROJECT_ROOT/.env"
AWS_REGION="ca-central-1"

# --- Step 1: Push API keys from local .env to SSM ---
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

push_key_to_ssm "GOOGLE_API_KEY"    "/openbrowser/GOOGLE_API_KEY"
push_key_to_ssm "OPENAI_API_KEY"    "/openbrowser/OPENAI_API_KEY"

# Also push GEMINI_API_KEY if present (some configs use this name)
GEMINI_KEY=$(grep "^GEMINI_API_KEY=" "$ENV_FILE" | head -1 | cut -d= -f2- || echo "")
if [ -n "$GEMINI_KEY" ] && [ "$GEMINI_KEY" != "PLACEHOLDER" ]; then
    GOOGLE_KEY=$(grep "^GOOGLE_API_KEY=" "$ENV_FILE" | head -1 | cut -d= -f2- || echo "")
    if [ -z "$GOOGLE_KEY" ]; then
        aws ssm put-parameter \
            --name "/openbrowser/GOOGLE_API_KEY" \
            --value "$GEMINI_KEY" \
            --type SecureString \
            --overwrite \
            --region "$AWS_REGION" \
            --output text > /dev/null 2>&1
        echo "  OK: GEMINI_API_KEY -> /openbrowser/GOOGLE_API_KEY"
    fi
fi

# --- Step 2: Read Terraform outputs ---
echo ""
echo "--- Reading Terraform outputs ---"
cd "$TF_DIR"

LAUNCH_TEMPLATE_ID=$(terraform output -raw launch_template_id 2>/dev/null)
RESULTS_BUCKET=$(terraform output -raw results_bucket 2>/dev/null)

if [ -z "$LAUNCH_TEMPLATE_ID" ]; then
    echo "ERROR: Could not read launch_template_id. Run 'terraform apply' first."
    exit 1
fi

echo "  Launch Template: $LAUNCH_TEMPLATE_ID"
echo "  Results Bucket:  $RESULTS_BUCKET"

# --- Step 3: Launch spot instance ---
echo ""
echo "--- Launching spot instance ---"

INSTANCE_ID=$(aws ec2 run-instances \
    --launch-template "LaunchTemplateId=$LAUNCH_TEMPLATE_ID,Version=\$Latest" \
    --count 1 \
    --region "$AWS_REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "  Instance ID: $INSTANCE_ID"

# --- Step 4: Wait for instance to be running ---
echo "  Waiting for instance to enter 'running' state..."
aws ec2 wait instance-running \
    --instance-ids "$INSTANCE_ID" \
    --region "$AWS_REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$AWS_REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "  Instance running at: $PUBLIC_IP"

# --- Step 5: Print monitoring instructions ---
echo ""
echo "=========================================="
echo "  Eval instance launched successfully"
echo "=========================================="
echo ""
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo "Region:      $AWS_REGION"
echo ""
echo "Monitor bootstrap progress:"
echo "  ssh ubuntu@$PUBLIC_IP 'tail -f /var/log/user-data.log'"
echo ""
echo "Monitor eval progress (after bootstrap completes):"
echo "  ssh ubuntu@$PUBLIC_IP 'tail -f /var/log/eval_run.log'"
echo ""
echo "SSM Session Manager (no SSH key needed):"
echo "  aws ssm start-session --target $INSTANCE_ID --region $AWS_REGION"
echo ""
echo "Results bucket: s3://$RESULTS_BUCKET/"
echo "Download results:"
echo "  uv run infra/eval/scripts/download_results.py --bucket $RESULTS_BUCKET"
echo ""
echo "Instance will auto-stop after 30 min idle."
echo "To terminate manually:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $AWS_REGION"
