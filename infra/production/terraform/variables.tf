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
}
