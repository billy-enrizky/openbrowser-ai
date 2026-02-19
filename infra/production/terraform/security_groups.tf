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

# CloudFront managed prefix list for restricting ALB ingress
data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

# ALB: allow HTTP from CloudFront (VNC WebSocket) and VPC (API Gateway VPC Link)
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "ALB: HTTP from CloudFront and VPC Link"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP from VPC (API Gateway VPC Link)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  ingress {
    description     = "HTTP from CloudFront (VNC WebSocket)"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront.id]
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
