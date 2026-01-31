# -----------------------------------------------------------------------------
# Security groups
# -----------------------------------------------------------------------------

# Backend EC2: allow ALB and VPC Link (API Gateway) to reach the app
resource "aws_security_group" "backend" {
  name        = "${var.project_name}-backend-sg"
  description = "Backend EC2: ALB and VPC Link traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Backend port from VPC (ALB and VPC Link)"
    from_port   = var.backend_port
    to_port     = var.backend_port
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    description = "All outbound (NAT for internet, VPC for DynamoDB/Secrets)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-backend-sg" }
}

# ALB: allow HTTP from anywhere (API Gateway and optional direct access)
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "ALB: HTTP from API Gateway and internet"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "To backend"
    from_port   = var.backend_port
    to_port     = var.backend_port
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = { Name = "${var.project_name}-alb-sg" }
}
