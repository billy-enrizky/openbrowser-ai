#!/bin/bash
set -euxo pipefail

# Install Docker
dnf update -y
dnf install -y docker
systemctl enable docker
systemctl start docker

# Optional: install jq for parsing Secrets Manager JSON
if [ -n "${secrets_secret_id}" ]; then
  dnf install -y jq
fi

# Env file for backend (DynamoDB, region, VNC, etc.)
cat > /opt/backend.env << 'ENVFILE'
${backend_env}
ENVFILE

# Fetch LLM API keys from SSM Parameter Store (encrypted with CMK)
fetch_ssm_key() {
  local param_name="$1"
  local env_var="$2"
  local val
  val=$(aws ssm get-parameter \
    --name "$param_name" \
    --with-decryption \
    --region "${aws_region}" \
    --query "Parameter.Value" \
    --output text 2>/dev/null || echo "")
  if [ -n "$val" ] && [ "$val" != "PLACEHOLDER" ]; then
    echo "$${env_var}=$${val}" >> /opt/backend.env
    echo "Loaded $env_var from SSM"
  else
    echo "Skipping $env_var (not set or placeholder)"
  fi
}

fetch_ssm_key "${ssm_google_key}"    "GOOGLE_API_KEY"
fetch_ssm_key "${ssm_openai_key}"    "OPENAI_API_KEY"
fetch_ssm_key "${ssm_anthropic_key}" "ANTHROPIC_API_KEY"

# Fallback: if Secrets Manager secret is set, fetch and append any missing keys
if [ -n "${secrets_secret_id}" ]; then
  echo "Checking Secrets Manager for additional keys..."
  SM_JSON=$(aws secretsmanager get-secret-value \
    --secret-id "${secrets_secret_id}" \
    --query SecretString --output text 2>/dev/null || echo "{}")
  if [ "$SM_JSON" != "{}" ]; then
    echo "$SM_JSON" | jq -r 'to_entries[] | select(.value != "replace-me" and .value != "PLACEHOLDER" and .value != "") | "\(.key)=\(.value)"' >> /opt/backend.env
    echo "Loaded additional keys from Secrets Manager"
  fi
fi

# Secure the env file
chmod 600 /opt/backend.env

# Pull and run backend container (Playwright needs 2GB shm and relaxed seccomp)
docker pull ${backend_image}

docker run -d --restart=always \
  --name openbrowser-backend \
  -p ${backend_port}:${backend_port} \
  --env-file /opt/backend.env \
  --shm-size=2g \
  --security-opt seccomp=unconfined \
  ${backend_image}
