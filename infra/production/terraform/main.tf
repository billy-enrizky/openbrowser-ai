# OpenBrowser-AI Production Infrastructure (Terraform root)
#
# NOTE:
# - Core infrastructure (VPC, ALB, backend EC2, API Gateway, Cognito, DynamoDB,
#   ECR, Secrets Manager, IAM, etc.) is defined in the other *.tf files in
#   this directory:
#     - versions.tf
#     - vpc.tf
#     - security_groups.tf
#     - alb.tf
#     - backend.tf
#     - api_gateway.tf
#     - iam.tf
#     - ecr.tf
#     - dynamodb.tf
#     - secrets.tf
# - This file intentionally only contains shared SSM parameters for API keys.

# ============================================================
# SSM PARAMETERS (API Keys)
# ============================================================

resource "aws_ssm_parameter" "google_api_key" {
  name      = "/${var.project_name}/GOOGLE_API_KEY"
  type      = "SecureString"
  value     = "PLACEHOLDER"
  overwrite = true

  lifecycle { ignore_changes = [value] }
  tags = { Name = "${var.project_name}-google-api-key" }
}

resource "aws_ssm_parameter" "openai_api_key" {
  name      = "/${var.project_name}/OPENAI_API_KEY"
  type      = "SecureString"
  value     = "PLACEHOLDER"
  overwrite = true

  lifecycle { ignore_changes = [value] }
  tags = { Name = "${var.project_name}-openai-api-key" }
}

resource "aws_ssm_parameter" "anthropic_api_key" {
  name      = "/${var.project_name}/ANTHROPIC_API_KEY"
  type      = "SecureString"
  value     = "PLACEHOLDER"
  overwrite = true

  lifecycle { ignore_changes = [value] }
  tags = { Name = "${var.project_name}-anthropic-api-key" }
}

