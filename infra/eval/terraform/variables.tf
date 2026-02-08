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

variable "instance_type" {
  description = "EC2 instance type for eval runner"
  type        = string
  default     = "t3.small"
}

variable "key_pair_name" {
  description = "SSH key pair name for eval EC2"
  type        = string
  default     = ""
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed for SSH access"
  type        = string
  default     = "0.0.0.0/0"
}
