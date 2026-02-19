#!/bin/bash
set -euo pipefail

# OpenBrowser-AI Production Deployment
# Deploys dockerized application to EC2
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

echo "=== OpenBrowser Production Bootstrap ==="
echo "Region: ${aws_region}"

# --- System Updates ---
apt-get update -y
apt-get upgrade -y

# --- Install Docker ---
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# --- Install AWS CLI v2 ---
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    apt-get install -y unzip
    unzip awscliv2.zip
    ./aws/install
    rm -rf aws awscliv2.zip
fi

# --- Clone Repository ---
cd /home/ubuntu
if [ ! -d "openbrowser-ai" ]; then
    git clone -b ${github_branch} ${github_repo_url} openbrowser-ai
fi
cd openbrowser-ai

# --- Get API Keys from SSM ---
GOOGLE_API_KEY=$(aws ssm get-parameter --name "/${project_name}/GOOGLE_API_KEY" --with-decryption --region ${aws_region} --query "Parameter.Value" --output text 2>/dev/null || echo "")
OPENAI_API_KEY=$(aws ssm get-parameter --name "/${project_name}/OPENAI_API_KEY" --with-decryption --region ${aws_region} --query "Parameter.Value" --output text 2>/dev/null || echo "")
ANTHROPIC_API_KEY=$(aws ssm get-parameter --name "/${project_name}/ANTHROPIC_API_KEY" --with-decryption --region ${aws_region} --query "Parameter.Value" --output text 2>/dev/null || echo "")

# Create .env file
cat > .env << ENVEOF
GOOGLE_API_KEY=${GOOGLE_API_KEY}
OPENAI_API_KEY=${OPENAI_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
VNC_ENABLED=${enable_vnc}
VNC_WIDTH=1920
VNC_HEIGHT=1080
DEBUG=false
CORS_ORIGINS=["*"]
ENVEOF

chmod 600 .env
chown ubuntu:ubuntu .env

# --- Create production docker-compose.yml ---
cat > docker-compose.prod.yml << PRODCOMPOSEEOF
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
      - "6080-6090:6080-6090"
    env_file:
      - .env
    restart: unless-stopped
    shm_size: '2gb'
    security_opt:
      - seccomp:unconfined
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
      - NEXT_PUBLIC_VNC_ENABLED=${enable_vnc}
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped

networks:
  default:
    name: openbrowser-network
PRODCOMPOSEEOF

chown -R ubuntu:ubuntu /home/ubuntu/openbrowser-ai

# --- Start Docker Compose ---
echo "=== Starting Docker Compose ==="
cd /home/ubuntu/openbrowser-ai
docker compose -f docker-compose.prod.yml up -d --build

# --- Wait for services to be healthy ---
echo "=== Waiting for services to be healthy ==="
sleep 10

# Check backend health
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend is healthy"
        break
    fi
    echo "Waiting for backend... ($i/30)"
    sleep 10
done

# Check frontend health
for i in {1..30}; do
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        echo "Frontend is healthy"
        break
    fi
    echo "Waiting for frontend... ($i/30)"
    sleep 10
done

echo "=== Deployment complete ==="
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Health: http://localhost:8000/health"
