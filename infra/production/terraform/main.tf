# OpenBrowser-AI Production Infrastructure
# EC2 + Docker + API Gateway deployment
#
# Usage:
#   cd infra/production/terraform
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
      Environment = "production"
      ManagedBy   = "terraform"
    }
  }
}

# ============================================================
# DATA SOURCES
# ============================================================

data "aws_region" "current" {}
data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

# ============================================================
# VPC + NETWORKING
# ============================================================

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${var.project_name}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

# Public subnets for ALB and EC2
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch  = true

  tags = { Name = "${var.project_name}-public-${count.index + 1}" }
}

# Private subnets for EC2 (optional - can use public for simplicity)
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 2)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = { Name = "${var.project_name}-private-${count.index + 1}" }
}

# Route table for public subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${var.project_name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# NAT Gateway for private subnets (optional - comment out if not needed)
resource "aws_eip" "nat" {
  count  = var.enable_nat ? 1 : 0
  domain = "vpc"
  tags   = { Name = "${var.project_name}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  count         = var.enable_nat ? 1 : 0
  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id
  tags          = { Name = "${var.project_name}-nat" }
}

resource "aws_route_table" "private" {
  count  = var.enable_nat ? 1 : 0
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main[0].id
  }

  tags = { Name = "${var.project_name}-private-rt" }
}

resource "aws_route_table_association" "private" {
  count          = var.enable_nat ? length(aws_subnet.private) : 0
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[0].id
}

# Security Group for ALB
resource "aws_security_group" "alb" {
  name_prefix = "${var.project_name}-alb-"
  description = "Security group for Application Load Balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP"
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = { Name = "${var.project_name}-alb-sg" }

  lifecycle { create_before_destroy = true }
}

# Security Group for EC2 instances
resource "aws_security_group" "ec2" {
  name_prefix = "${var.project_name}-ec2-"
  description = "Security group for EC2 instances running Docker"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "Backend API from ALB"
  }

  ingress {
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "Frontend from ALB"
  }

  ingress {
    from_port       = 6080
    to_port         = 6090
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
    description     = "VNC websockify from ALB"
  }

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

  tags = { Name = "${var.project_name}-ec2-sg" }

  lifecycle { create_before_destroy = true }
}

# ============================================================
# APPLICATION LOAD BALANCER
# ============================================================

resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false
  enable_http2               = true
  enable_cross_zone_load_balancing = true

  tags = { Name = "${var.project_name}-alb" }
}

resource "aws_lb_target_group" "backend" {
  name     = "${var.project_name}-backend-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/health"
    matcher             = "200"
  }

  deregistration_delay = 30

  tags = { Name = "${var.project_name}-backend-tg" }
}

resource "aws_lb_target_group" "frontend" {
  name     = "${var.project_name}-frontend-tg"
  port     = 3000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 30
    path                = "/"
    matcher             = "200"
  }

  deregistration_delay = 30

  tags = { Name = "${var.project_name}-frontend-tg" }
}

# ALB Listener - HTTP with path-based routing
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  # Route /ws and /ws/* to backend (WebSocket) - evaluated first
  rule {
    priority = 100
    action {
      type             = "forward"
      target_group_arn = aws_lb_target_group.backend.arn
    }
    condition {
      path_pattern {
        values = ["/ws", "/ws/*"]
      }
    }
  }

  # Route /api/* to backend
  rule {
    priority = 200
    action {
      type             = "forward"
      target_group_arn = aws_lb_target_group.backend.arn
    }
    condition {
      path_pattern {
        values = ["/api/*"]
      }
    }
  }

  # Route /health to backend
  rule {
    priority = 300
    action {
      type             = "forward"
      target_group_arn = aws_lb_target_group.backend.arn
    }
    condition {
      path_pattern {
        values = ["/health"]
      }
    }
  }

  # Default action: everything else goes to frontend
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
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

resource "aws_iam_role" "ec2" {
  name               = "${var.project_name}-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = { Name = "${var.project_name}-ec2-role" }
}

resource "aws_iam_instance_profile" "ec2" {
  name = "${var.project_name}-ec2-profile"
  role = aws_iam_role.ec2.name
}

# EC2 permissions for SSM, CloudWatch, and ECR (if using ECR)
data "aws_iam_policy_document" "ec2_permissions" {
  # SSM for secure access
  statement {
    actions   = ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"]
    resources = ["arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/${var.project_name}/*"]
  }

  # CloudWatch Logs
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:*"]
  }

  # ECR (if using container registry)
  statement {
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "ec2" {
  name   = "${var.project_name}-ec2-policy"
  role   = aws_iam_role.ec2.id
  policy = data.aws_iam_policy_document.ec2_permissions.json
}

resource "aws_iam_role_policy_attachment" "ec2_ssm" {
  role       = aws_iam_role.ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# ============================================================
# EC2 LAUNCH TEMPLATE
# ============================================================

resource "aws_launch_template" "app" {
  name_prefix            = "${var.project_name}-app-"
  image_id               = var.ubuntu_ami_id
  instance_type          = var.instance_type
  update_default_version = true
  key_name               = var.key_pair_name != "" ? var.key_pair_name : null

  iam_instance_profile {
    name = aws_iam_instance_profile.ec2.name
  }

  network_interfaces {
    associate_public_ip_address = true
    security_groups             = [aws_security_group.ec2.id]
    subnet_id                   = aws_subnet.public[0].id
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = var.ebs_volume_size
      volume_type = "gp3"
      encrypted   = true
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    project_name        = var.project_name
    aws_region          = var.aws_region
    docker_image_tag    = var.docker_image_tag
    github_repo_url     = var.github_repo_url
    github_branch       = var.github_branch
    enable_vnc          = var.enable_vnc
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "${var.project_name}-app"
      Role    = "app"
      Project = var.project_name
    }
  }

  tags = { Name = "${var.project_name}-app-template" }
}

# ============================================================
# AUTO SCALING GROUP
# ============================================================

resource "aws_autoscaling_group" "app" {
  name                = "${var.project_name}-asg"
  vpc_zone_identifier = aws_subnet.public[*].id
  target_group_arns    = [aws_lb_target_group.backend.arn, aws_lb_target_group.frontend.arn]
  health_check_type    = "ELB"
  health_check_grace_period = 300

  min_size         = var.min_instances
  max_size         = var.max_instances
  desired_capacity = var.desired_instances

  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-app"
    propagate_at_launch = true
  }

  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 50
    }
  }
}

# ============================================================
# API GATEWAY v2 (HTTP API)
# ============================================================

resource "aws_apigatewayv2_api" "main" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"
  description   = "API Gateway for OpenBrowser-AI"

  cors_configuration {
    allow_origins = var.cors_origins
    allow_methods = ["*"]
    allow_headers = ["*"]
    allow_credentials = true
    max_age = 300
  }

  tags = { Name = "${var.project_name}-api" }
}

# Integration with ALB (HTTP_PROXY - direct connection)
# Note: For production, consider using VPC Link with NLB for better security
resource "aws_apigatewayv2_integration" "alb" {
  api_id           = aws_apigatewayv2_api.main.id
  integration_type = "HTTP_PROXY"
  integration_uri  = "http://${aws_lb.main.dns_name}"
  integration_method = "ANY"

  payload_format_version = "1.0"
  
  # Connection timeout
  timeout_milliseconds = 30000
}

# Default route - proxy all requests to ALB
resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.alb.id}"
}

# Note: WebSocket routes are handled by the $default route
# API Gateway HTTP API supports WebSocket through the same integration

# Stage
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true

  default_route_settings {
    detailed_metrics_enabled = true
    logging_level            = "INFO"
  }

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }
}

# CloudWatch Log Group for API Gateway
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${var.project_name}"
  retention_in_days = 7
  tags              = { Name = "${var.project_name}-api-logs" }
}

# ============================================================
# SSM PARAMETERS (API Keys)
# ============================================================

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

resource "aws_ssm_parameter" "anthropic_api_key" {
  name  = "/${var.project_name}/ANTHROPIC_API_KEY"
  type  = "SecureString"
  value = "PLACEHOLDER"

  lifecycle { ignore_changes = [value] }
  tags = { Name = "${var.project_name}-anthropic-api-key" }
}
