# KMS CMK for encrypting auth profile data (cookies/localStorage)
# Separate from the SSM key in kms.tf -- this key is for direct envelope encryption
# Note: data "aws_caller_identity" "current" {} already exists in versions.tf -- do NOT duplicate

resource "aws_kms_key" "auth_data" {
  description             = "Encrypts saved browser auth state for OpenBrowser"
  deletion_window_in_days = 14
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "RootAdmin"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "BackendEnvelopeEncryption"
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.backend.arn
        }
        Action = [
          "kms:GenerateDataKey",
          "kms:Decrypt",
          "kms:DescribeKey",
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:EncryptionContext:purpose" = "auth_profile"
          }
        }
      },
    ]
  })

  tags = { Name = "${var.project_name}-auth-data-cmk" }
}

resource "aws_kms_alias" "auth_data" {
  name          = "alias/${var.project_name}-auth-data"
  target_key_id = aws_kms_key.auth_data.key_id
}
