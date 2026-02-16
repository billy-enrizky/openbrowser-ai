# OpenBrowser-AI Evaluation Infrastructure
# Self-contained Terraform config: VPC + S3 + IAM + EC2 Spot
#
# Usage:
#   cd infra/eval/terraform
#   terraform init
#   terraform plan
#   terraform apply

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = "eval"
      ManagedBy   = "terraform"
    }
  }
}

# ============================================================
# DATA SOURCES
# ============================================================

data "aws_region" "current" {}

data "aws_caller_identity" "current" {}

# Ubuntu 22.04 LTS AMI (hardcoded -- ec2:DescribeImages blocked by org SCP)
# ca-central-1, amd64, hvm:ebs-ssd, 20251212 release
# Source: https://cloud-images.ubuntu.com/locator/ec2/

# ============================================================
# VPC + NETWORKING
# ============================================================

resource "aws_vpc" "eval" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${var.project_name}-eval-vpc" }
}

resource "aws_internet_gateway" "eval" {
  vpc_id = aws_vpc.eval.id
  tags   = { Name = "${var.project_name}-eval-igw" }
}

resource "aws_subnet" "eval_public" {
  vpc_id                  = aws_vpc.eval.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = { Name = "${var.project_name}-eval-public" }
}

resource "aws_route_table" "eval_public" {
  vpc_id = aws_vpc.eval.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.eval.id
  }

  tags = { Name = "${var.project_name}-eval-public-rt" }
}

resource "aws_route_table_association" "eval_public" {
  subnet_id      = aws_subnet.eval_public.id
  route_table_id = aws_route_table.eval_public.id
}

resource "aws_security_group" "eval" {
  name_prefix = "${var.project_name}-eval-"
  description = "Security group for eval EC2 spot instance"
  vpc_id      = aws_vpc.eval.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
    description = "SSH"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = { Name = "${var.project_name}-eval-sg" }

  lifecycle { create_before_destroy = true }
}

# ============================================================
# S3 BUCKETS
# ============================================================

# Dataset bucket: mind2web/, webarena/, formfactory/, stress-tests/
resource "aws_s3_bucket" "datasets" {
  bucket        = "${var.project_name}-eval-datasets-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
  tags          = { Name = "${var.project_name}-eval-datasets" }
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket                  = aws_s3_bucket.datasets.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  rule {
    id     = "transition-to-ia"
    status = "Enabled"
    filter {}
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}

# Results bucket: {project}/runs/{ISO_DATE}/{run_id}/
resource "aws_s3_bucket" "results" {
  bucket = "${var.project_name}-eval-results-${data.aws_caller_identity.current.account_id}"
  tags   = { Name = "${var.project_name}-eval-results" }
}

resource "aws_s3_bucket_versioning" "results" {
  bucket = aws_s3_bucket.results.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "results" {
  bucket                  = aws_s3_bucket.results.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "results" {
  bucket = aws_s3_bucket.results.id
  rule {
    id     = "expire-old-results"
    status = "Enabled"
    filter {}
    expiration { days = 90 }
  }
}

# ============================================================
# IAM
# ============================================================

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eval" {
  name               = "${var.project_name}-eval-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = { Name = "${var.project_name}-eval-role" }
}

resource "aws_iam_instance_profile" "eval" {
  name = "${var.project_name}-eval-profile"
  role = aws_iam_role.eval.name
}

data "aws_iam_policy_document" "eval_permissions" {
  # Read datasets
  statement {
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.datasets.arn, "${aws_s3_bucket.datasets.arn}/*"]
  }
  # Write results
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
    resources = [aws_s3_bucket.results.arn, "${aws_s3_bucket.results.arn}/*"]
  }
  # CloudWatch Logs
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:*"]
  }
  # SSM read (API keys)
  statement {
    actions   = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"]
    resources = ["arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/${var.project_name}/*"]
  }
  # EC2 self-stop
  statement {
    actions   = ["ec2:StopInstances", "ec2:DescribeInstances", "ec2:DescribeTags"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "eval" {
  name   = "${var.project_name}-eval-policy"
  role   = aws_iam_role.eval.id
  policy = data.aws_iam_policy_document.eval_permissions.json
}

resource "aws_iam_role_policy_attachment" "eval_ssm" {
  role       = aws_iam_role.eval.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# ============================================================
# EC2 SPOT LAUNCH TEMPLATE
# ============================================================

resource "aws_launch_template" "eval" {
  name_prefix            = "${var.project_name}-eval-"
  image_id               = var.ubuntu_ami_id
  instance_type          = var.instance_type
  update_default_version = true
  key_name      = var.key_pair_name != "" ? var.key_pair_name : null

  iam_instance_profile {
    name = aws_iam_instance_profile.eval.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.eval.id]
    subnet_id                   = aws_subnet.eval_public.id
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 30
      volume_type = "gp3"
      encrypted   = true
    }
  }

  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price          = "0.02"
      spot_instance_type = "one-time"
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    project_name        = var.project_name
    aws_region          = var.aws_region
    data_bucket_name    = aws_s3_bucket.datasets.id
    results_bucket_name = aws_s3_bucket.results.id
    eval_datasets       = var.eval_datasets
    eval_max_tasks      = var.eval_max_tasks
    eval_models         = var.eval_models
    eval_agent_types    = var.eval_agent_types
    auto_run_eval       = var.auto_run_eval
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "${var.project_name}-eval-runner"
      Role    = "eval"
      Project = var.project_name
    }
  }

  tags = { Name = "${var.project_name}-eval-template" }
}

# ============================================================
# SSM PARAMETERS (API Keys)
# ============================================================
# Created as placeholders -- real values are pushed by launch_eval.sh
# from the local .env file before launching an instance.

resource "aws_ssm_parameter" "google_api_key" {
  name  = "/${var.project_name}/GOOGLE_API_KEY"
  type  = "SecureString"
  value = "PLACEHOLDER"

  lifecycle { ignore_changes = [value] }
  tags = { Name = "${var.project_name}-google-api-key" }
}

resource "aws_ssm_parameter" "openai_api_key" {
  name  = "/${var.project_name}/OPENAI_API_KEY"
  type  = "SecureString"
  value = "PLACEHOLDER"

  lifecycle { ignore_changes = [value] }
  tags = { Name = "${var.project_name}-openai-api-key" }
}

