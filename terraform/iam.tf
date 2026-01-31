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

# Secrets Manager read (for LLM API keys) - when using our secret or an external one
data "aws_iam_policy_document" "backend_secrets" {
  count = (var.secrets_manager_secret_name != "" || length(aws_secretsmanager_secret.backend_keys) > 0) ? 1 : 0

  statement {
    sid    = "SecretsManagerRead"
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue"
    ]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "secretsmanager:VersionStage"
      values   = ["AWSCURRENT"]
    }
  }
}

resource "aws_iam_policy" "backend_secrets" {
  count = (var.secrets_manager_secret_name != "" || length(aws_secretsmanager_secret.backend_keys) > 0) ? 1 : 0

  name   = "${var.project_name}-backend-secrets"
  policy = data.aws_iam_policy_document.backend_secrets[0].json
}

resource "aws_iam_role_policy_attachment" "backend_secrets" {
  count = (var.secrets_manager_secret_name != "" || length(aws_secretsmanager_secret.backend_keys) > 0) ? 1 : 0

  role       = aws_iam_role.backend.name
  policy_arn = aws_iam_policy.backend_secrets[0].arn
}

resource "aws_iam_instance_profile" "backend" {
  name = "${var.project_name}-backend-profile"
  role = aws_iam_role.backend.name
}
