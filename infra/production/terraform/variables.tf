## -----------------------------------------------------------------------------
## General
## -----------------------------------------------------------------------------

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

## -----------------------------------------------------------------------------
## Networking
## -----------------------------------------------------------------------------

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

## -----------------------------------------------------------------------------
## Backend (EC2 + API)
## -----------------------------------------------------------------------------

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
  default     = "t3.medium"
  description = "EC2 instance type for the backend (needs 4GB+ memory for Chromium/Playwright)."
}

variable "secrets_manager_secret_name" {
  type        = string
  default     = ""
  description = "Name or ARN of Secrets Manager secret containing LLM API keys. Leave empty if not using secrets yet."
}

variable "postgres_db_name" {
  type        = string
  default     = "openbrowser"
  description = "PostgreSQL database name."
}

variable "postgres_username" {
  type        = string
  default     = "openbrowser"
  description = "Master username for PostgreSQL."
}

variable "postgres_instance_class" {
  type        = string
  default     = "db.t4g.micro"
  description = "RDS instance class for PostgreSQL."
}

variable "postgres_allocated_storage" {
  type        = number
  default     = 20
  description = "Allocated storage (GiB) for PostgreSQL."
}

variable "postgres_backup_retention_days" {
  type        = number
  default     = 7
  description = "Backup retention period for PostgreSQL."
}

variable "postgres_skip_final_snapshot" {
  type        = bool
  default     = true
  description = "Skip final snapshot on destroy (set false for production safety)."
}

## -----------------------------------------------------------------------------
## Frontend (S3 + CloudFront)
## -----------------------------------------------------------------------------

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

## -----------------------------------------------------------------------------
## Cognito authentication
## -----------------------------------------------------------------------------

variable "enable_backend_auth" {
  type        = bool
  default     = true
  description = "Enable Cognito JWT verification in the backend service (REST + WebSocket)."
}

variable "cognito_domain_prefix" {
  type        = string
  default     = ""
  description = "Prefix for Cognito hosted UI domain. If empty, a random suffix is used."
}

variable "cognito_callback_urls" {
  type        = list(string)
  default     = []
  description = "Allowed Cognito callback URLs. Leave empty to auto-generate from frontend domain + localhost."
}

variable "cognito_logout_urls" {
  type        = list(string)
  default     = []
  description = "Allowed Cognito logout URLs. Leave empty to auto-generate from frontend domain + localhost."
}

variable "cognito_oauth_scopes" {
  type        = list(string)
  default     = ["openid", "email", "profile"]
  description = "OAuth scopes allowed on the Cognito app client."
}

## -----------------------------------------------------------------------------
## CORS
## -----------------------------------------------------------------------------

variable "cors_origins" {
  description = "Allowed CORS origins"
  type        = list(string)
  default     = ["*"]
}
