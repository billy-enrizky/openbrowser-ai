# -----------------------------------------------------------------------------
# KMS Customer Managed Key for encrypting backend API keys in SSM
# Only the backend EC2 role can decrypt; other account users cannot read keys.
# -----------------------------------------------------------------------------

resource "aws_kms_key" "api_keys" {
  description             = "Encrypts LLM API keys in SSM for OpenBrowser backend"
  deletion_window_in_days = 14
  enable_key_rotation     = true

  # Key policy: root for management, backend role for decrypt via SSM only
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # 1. Account root -- required for key administration via IAM policies
      {
        Sid    = "RootAdmin"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      # 2. Backend EC2 role -- decrypt only, and only when called through SSM
      {
        Sid    = "BackendDecryptViaSSM"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.backend.arn
        }
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "ssm.${var.aws_region}.amazonaws.com"
          }
        }
      },
    ]
  })

  tags = { Name = "${var.project_name}-api-keys-cmk" }
}

resource "aws_kms_alias" "api_keys" {
  name          = "alias/${var.project_name}-api-keys"
  target_key_id = aws_kms_key.api_keys.key_id
}
