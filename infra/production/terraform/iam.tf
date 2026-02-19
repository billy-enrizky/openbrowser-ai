# -----------------------------------------------------------------------------
# IAM role for backend EC2: DynamoDB, SSM, ECR, Secrets Manager
# -----------------------------------------------------------------------------

data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "backend" {
  name               = "${var.project_name}-backend-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

# DynamoDB access for session/chats
data "aws_iam_policy_document" "backend_dynamodb" {
  statement {
    sid = "DynamoDBAccess"
    actions = [
      "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem",
      "dynamodb:Query", "dynamodb:Scan", "dynamodb:BatchGetItem", "dynamodb:BatchWriteItem",
      "dynamodb:DescribeTable", "dynamodb:ConditionCheckItem"
    ]
    resources = [aws_dynamodb_table.main.arn, "${aws_dynamodb_table.main.arn}/index/*"]
  }
}

resource "aws_iam_policy" "backend_dynamodb" {
  name   = "${var.project_name}-backend-dynamodb"
  policy = data.aws_iam_policy_document.backend_dynamodb.json
}

# SSM for Session Manager (no SSH needed)
resource "aws_iam_role_policy_attachment" "backend_ssm" {
  role       = aws_iam_role.backend.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# ECR pull for backend image
resource "aws_iam_role_policy_attachment" "backend_ecr" {
  role       = aws_iam_role.backend.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "backend_dynamodb" {
  role       = aws_iam_role.backend.name
  policy_arn = aws_iam_policy.backend_dynamodb.arn
}

# Secrets Manager read (for LLM API keys) - scoped to the specific secret ARN
data "aws_iam_policy_document" "backend_secrets" {
  count = 1

  statement {
    sid    = "SecretsManagerRead"
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue"
    ]
    resources = length(aws_secretsmanager_secret.backend_keys) > 0 ? [aws_secretsmanager_secret.backend_keys[0].arn] : []
    condition {
      test     = "StringEquals"
      variable = "secretsmanager:VersionStage"
      values   = ["AWSCURRENT"]
    }
  }
}

resource "aws_iam_policy" "backend_secrets" {
  count = 1

  name   = "${var.project_name}-backend-secrets"
  policy = data.aws_iam_policy_document.backend_secrets[0].json
}

resource "aws_iam_role_policy_attachment" "backend_secrets" {
  count = 1

  role       = aws_iam_role.backend.name
  policy_arn = aws_iam_policy.backend_secrets[0].arn
}

# SSM Parameter Store read (for LLM API keys encrypted with CMK)
data "aws_iam_policy_document" "backend_ssm_keys" {
  statement {
    sid    = "SSMGetAPIKeys"
    effect = "Allow"
    actions = [
      "ssm:GetParameter",
      "ssm:GetParameters",
    ]
    resources = [
      aws_ssm_parameter.google_api_key.arn,
      aws_ssm_parameter.openai_api_key.arn,
      aws_ssm_parameter.anthropic_api_key.arn,
    ]
  }
}

resource "aws_iam_policy" "backend_ssm_keys" {
  name   = "${var.project_name}-backend-ssm-keys"
  policy = data.aws_iam_policy_document.backend_ssm_keys.json
}

resource "aws_iam_role_policy_attachment" "backend_ssm_keys" {
  role       = aws_iam_role.backend.name
  policy_arn = aws_iam_policy.backend_ssm_keys.arn
}

resource "aws_iam_instance_profile" "backend" {
  name = "${var.project_name}-backend-profile"
  role = aws_iam_role.backend.name
}
