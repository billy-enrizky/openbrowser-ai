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
