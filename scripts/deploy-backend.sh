#!/usr/bin/env bash
set -euxo pipefail

# Backend deployment script for OpenBrowser-AI production
# - Ensures ECR repo exists
# - Builds and pushes backend Docker image
# - Applies full Terraform infra (VPC, ALB, EC2 backend, API Gateway, etc.)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="$ROOT_DIR/infra/production/terraform"
REGION="ca-central-1"

cd "$TF_DIR"

# 1) Ensure ECR repo exists (no-op if already created)
terraform init
terraform apply -target=aws_ecr_repository.backend

# 2) Get ECR repo URL from Terraform outputs
REPO="$(terraform output -raw backend_ecr_repository_url)"

# 3) Login to ECR
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${REPO%%/*}"

# 4) Build backend image from repo root
cd "$ROOT_DIR"
docker build -t openbrowser-backend -f backend/Dockerfile .

# 5) Tag + push (uses backend_image_tag in terraform.tfvars, typically 'latest')
docker tag openbrowser-backend:latest "$REPO:latest"
docker push "$REPO:latest"

# 6) Apply full infra (creates/updates VPC, ALB, EC2 backend, API Gateway, etc.)
cd "$TF_DIR"
terraform apply

