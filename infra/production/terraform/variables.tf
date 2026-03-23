variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "openbrowser"
}

variable "aws_region" {
  description = "AWS region for infrastructure"
  type        = string
  default     = "ca-central-1"
}

variable "backend_iam_role_name" {
  description = "Name of the existing backend ECS task IAM role to attach new policies to"
  type        = string
}
