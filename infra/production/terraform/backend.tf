# -----------------------------------------------------------------------------
# Backend EC2: Dockerized FastAPI + Playwright (private subnet)
# -----------------------------------------------------------------------------

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

locals {
  backend_env = [
    "AWS_REGION=${var.aws_region}",
    "DDB_TABLE=${aws_dynamodb_table.main.name}",
    "CORS_ORIGINS=${join(",", var.cors_origins)}",
    "VNC_ENABLED=true",
    "VNC_WIDTH=1920",
    "VNC_HEIGHT=1080",
    "DEBUG=false",
    "AUTH_ENABLED=${var.enable_backend_auth}",
    "COGNITO_REGION=${var.aws_region}",
    "COGNITO_USER_POOL_ID=${aws_cognito_user_pool.main.id}",
    "COGNITO_APP_CLIENT_ID=${aws_cognito_user_pool_client.app.id}",
  ]
  # Use provided secret name or the one we create
  backend_secret_id = var.secrets_manager_secret_name != "" ? var.secrets_manager_secret_name : (length(aws_secretsmanager_secret.backend_keys) > 0 ? aws_secretsmanager_secret.backend_keys[0].name : "")
  # Image: explicit URI or ECR repo (created by this module) + tag
  backend_image_uri = var.backend_image != "" ? var.backend_image : "${aws_ecr_repository.backend.repository_url}:${var.backend_image_tag}"
}

resource "aws_instance" "backend" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.backend_instance_type
  subnet_id              = aws_subnet.private[0].id
  vpc_security_group_ids = [aws_security_group.backend.id]
  iam_instance_profile   = aws_iam_instance_profile.backend.name

  associate_public_ip_address = false
  user_data_replace_on_change = true

  user_data = templatefile("${path.module}/scripts/backend-userdata.sh", {
    backend_image     = local.backend_image_uri
    backend_port      = var.backend_port
    backend_env       = join("\n", local.backend_env)
    secrets_secret_id = local.backend_secret_id
  })

  tags = { Name = "${var.project_name}-backend" }
}

resource "aws_lb_target_group_attachment" "backend" {
  target_group_arn = aws_lb_target_group.backend.arn
  target_id        = aws_instance.backend.id
  port             = var.backend_port
}
