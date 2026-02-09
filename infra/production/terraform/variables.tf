variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "openbrowser"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ca-central-1"
}

variable "ubuntu_ami_id" {
  description = "Ubuntu 22.04 LTS AMI ID"
  type        = string
  default     = "ami-0631168b8ae6e1731" # ca-central-1, amd64, hvm:ebs-ssd, 20251212
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium" # Minimum for Docker + browser automation
}

variable "key_pair_name" {
  description = "SSH key pair name for EC2"
  type        = string
  default     = ""
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed for SSH access"
  type        = string
  default     = "0.0.0.0/0"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_nat" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = false
}

variable "ebs_volume_size" {
  description = "EBS volume size in GB"
  type        = number
  default     = 50
}

# Auto Scaling
variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 3
}

variable "desired_instances" {
  description = "Desired number of instances"
  type        = number
  default     = 1
}

# Docker deployment
variable "docker_image_tag" {
  description = "Docker image tag to deploy (if using ECR) or 'latest' for git clone"
  type        = string
  default     = "latest"
}

variable "github_repo_url" {
  description = "GitHub repository URL to clone"
  type        = string
  default     = "https://github.com/billy-enrizky/openbrowser-ai.git"
}

variable "github_branch" {
  description = "GitHub branch to deploy"
  type        = string
  default     = "main"
}

variable "enable_vnc" {
  description = "Enable VNC for live browser viewing"
  type        = bool
  default     = true
}

# API Gateway
variable "cors_origins" {
  description = "Allowed CORS origins"
  type        = list(string)
  default     = ["*"]
# -----------------------------------------------------------------------------
# General
# -----------------------------------------------------------------------------
variable "aws_region" {
  type        = string
  default     = "ca-central-1"
  description = "AWS region for all resources."
}

variable "project_name" {
  type        = string
  default     = "openbrowser"
  description = "Project name used for resource naming and tags."
}

# -----------------------------------------------------------------------------
# Networking
# -----------------------------------------------------------------------------
variable "vpc_cidr" {
  type        = string
  default     = "10.0.0.0/16"
  description = "CIDR block for the VPC."
}

variable "public_subnet_cidrs" {
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
  description = "CIDR blocks for public subnets (one per AZ)."
}

variable "private_subnet_cidrs" {
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
  description = "CIDR blocks for private subnets (one per AZ)."
}

# -----------------------------------------------------------------------------
# Backend (EC2 + API)
# -----------------------------------------------------------------------------
variable "backend_port" {
  type        = number
  default     = 8000
  description = "Port the backend container listens on (FastAPI default 8000)."
}

variable "backend_image" {
  type        = string
  default     = ""
  description = "Backend container image URI (ECR or Docker Hub). If empty, the image is expected at the ECR repo created by this module (see ecr.tf) with tag from backend_image_tag."
}

variable "backend_image_tag" {
  type        = string
  default     = "latest"
  description = "Tag to use when pulling from the Terraform-created ECR repo (only when backend_image is empty)."
}

variable "backend_instance_type" {
  type        = string
  default     = "t3.small"
  description = "EC2 instance type for the backend (needs 2GB+ memory for Chromium/Playwright)."
}

variable "secrets_manager_secret_name" {
  type        = string
  default     = ""
  description = "Name or ARN of Secrets Manager secret containing LLM API keys. Leave empty if not using secrets yet."
}

# -----------------------------------------------------------------------------
# Frontend (S3 + CloudFront)
# -----------------------------------------------------------------------------
variable "frontend_domain_name" {
  type        = string
  default     = ""
  description = "Optional custom domain for the frontend (e.g. app.example.com). Leave empty to use CloudFront default domain."
}

variable "frontend_acm_certificate_arn" {
  type        = string
  default     = ""
  description = "ACM certificate ARN for frontend_domain_name (must be in us-east-1 for CloudFront)."
}

# -----------------------------------------------------------------------------
# Cognito (auth - for future use)
# -----------------------------------------------------------------------------
variable "enable_api_auth" {
  type        = bool
  default     = false
  description = "Require JWT (Cognito) on API routes. Set to true when auth is implemented."
}

variable "cognito_domain_prefix" {
  type        = string
  default     = ""
  description = "Prefix for Cognito hosted UI domain. If empty, a random suffix is used."
}
