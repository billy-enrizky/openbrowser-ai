#!/bin/bash
set -euo pipefail

# Eval EC2 bootstrap: Python 3.12 + uv + Playwright + Chromium + Xvfb
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

echo "=== OpenBrowser Eval Bootstrap ==="
echo "Region: ${aws_region}"

# --- System ---
apt-get update -y
apt-get upgrade -y
apt-get install -y \
    software-properties-common curl wget gnupg ca-certificates git jq unzip awscli \
    fonts-liberation libasound2 libatk-bridge2.0-0 libatk1.0-0 libatspi2.0-0 \
    libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 \
    libxcomposite1 libxdamage1 libxfixes3 libxkbcommon0 libxrandr2 xdg-utils xvfb

# --- Python 3.12 ---
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -y
apt-get install -y python3.12 python3.12-venv python3.12-dev

# --- uv ---
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# --- Clone repo ---
cd /home/ubuntu
if [ ! -d "openbrowser-ai" ]; then
    git clone https://github.com/billy-enrizky/openbrowser-ai.git
fi
cd openbrowser-ai

# --- Install ---
uv sync --all-extras
uv run playwright install chromium
uv run playwright install-deps chromium

# --- API keys from SSM ---
GOOGLE_API_KEY=$(aws ssm get-parameter --name "/${project_name}/GOOGLE_API_KEY" --with-decryption --region ${aws_region} --query "Parameter.Value" --output text 2>/dev/null || echo "")
OPENAI_API_KEY=$(aws ssm get-parameter --name "/${project_name}/OPENAI_API_KEY" --with-decryption --region ${aws_region} --query "Parameter.Value" --output text 2>/dev/null || echo "")
ANTHROPIC_API_KEY=$(aws ssm get-parameter --name "/${project_name}/ANTHROPIC_API_KEY" --with-decryption --region ${aws_region} --query "Parameter.Value" --output text 2>/dev/null || echo "")

cat > .env << ENVEOF
GOOGLE_API_KEY=$${GOOGLE_API_KEY}
OPENAI_API_KEY=$${OPENAI_API_KEY}
ANTHROPIC_API_KEY=$${ANTHROPIC_API_KEY}
DATA_BUCKET=${data_bucket_name}
RESULTS_BUCKET=${results_bucket_name}
ENVEOF

chmod 600 .env
chown -R ubuntu:ubuntu /home/ubuntu/openbrowser-ai

# --- Auto-stop cron (30 min idle) ---
cat > /home/ubuntu/eval_auto_stop.sh << 'STOPEOF'
#!/bin/bash
if ! pgrep -f "eval_benchmark" > /dev/null 2>&1; then
    IDLE_FILE="/tmp/eval_idle_since"
    if [ ! -f "$IDLE_FILE" ]; then
        date +%s > "$IDLE_FILE"
        exit 0
    fi
    IDLE_SINCE=$(cat "$IDLE_FILE")
    NOW=$(date +%s)
    if [ $((NOW - IDLE_SINCE)) -ge 1800 ]; then
        TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
        INSTANCE_ID=$(curl -s "http://169.254.169.254/latest/meta-data/instance-id" -H "X-aws-ec2-metadata-token: $TOKEN")
        REGION=$(curl -s "http://169.254.169.254/latest/meta-data/placement/region" -H "X-aws-ec2-metadata-token: $TOKEN")
        aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
    fi
else
    rm -f /tmp/eval_idle_since
fi
STOPEOF

chmod +x /home/ubuntu/eval_auto_stop.sh
chown ubuntu:ubuntu /home/ubuntu/eval_auto_stop.sh
(crontab -l -u ubuntu 2>/dev/null || echo "") | grep -v "eval_auto_stop" | { cat; echo "*/10 * * * * /home/ubuntu/eval_auto_stop.sh >> /var/log/eval_auto_stop.log 2>&1"; } | crontab -u ubuntu -

echo "=== Eval bootstrap complete ==="
