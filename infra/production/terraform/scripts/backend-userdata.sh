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

# If Secrets Manager secret is set, fetch and append (expect JSON: {"KEY":"value",...})
if [ -n "${secrets_secret_id}" ]; then
  aws secretsmanager get-secret-value \
    --secret-id "${secrets_secret_id}" \
    --query SecretString --output text \
  | jq -r 'to_entries | map("\(.key)=\(.value)") | .[]' >> /opt/backend.env
fi

# Pull and run backend container (Playwright needs 2GB shm and relaxed seccomp)
docker pull ${backend_image}

docker run -d --restart=always \
  --name openbrowser-backend \
  -p ${backend_port}:${backend_port} \
  --env-file /opt/backend.env \
  --shm-size=2g \
  --security-opt seccomp=unconfined \
  ${backend_image}
