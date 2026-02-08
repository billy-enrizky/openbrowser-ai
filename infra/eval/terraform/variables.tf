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
  description = "Ubuntu 22.04 LTS AMI ID (hardcoded -- ec2:DescribeImages blocked by org SCP)"
  type        = string
  default     = "ami-0631168b8ae6e1731" # ca-central-1, amd64, hvm:ebs-ssd, 20251212
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

# Eval run configuration (passed to user_data.sh)

variable "eval_datasets" {
  description = "Datasets to evaluate (space-separated)"
  type        = string
  default     = "formfactory"
}

variable "eval_max_tasks" {
  description = "Max tasks per dataset (0 = all)"
  type        = number
  default     = 2
}

variable "eval_models" {
  description = "LLM models to test (space-separated)"
  type        = string
  default     = "gemini-2.5-flash"
}

variable "eval_agent_types" {
  description = "Agent types to test (space-separated)"
  type        = string
  default     = "Agent"
}

variable "auto_run_eval" {
  description = "Automatically run eval on instance boot"
  type        = bool
  default     = true
}
