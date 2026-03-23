# KMS for auth profile encryption
data "aws_iam_policy_document" "backend_kms_auth" {
  statement {
    sid    = "KMSAuthEncryption"
    effect = "Allow"
    actions = [
      "kms:GenerateDataKey",
      "kms:Decrypt",
      "kms:DescribeKey",
    ]
    resources = [aws_kms_key.auth_data.arn]
  }
}

resource "aws_iam_policy" "backend_kms_auth" {
  name   = "${var.project_name}-backend-kms-auth"
  policy = data.aws_iam_policy_document.backend_kms_auth.json
}

resource "aws_iam_role_policy_attachment" "backend_kms_auth" {
  role       = aws_iam_role.backend.name
  policy_arn = aws_iam_policy.backend_kms_auth.arn
}

# SQS for scheduled job messages
data "aws_iam_policy_document" "backend_sqs" {
  statement {
    sid    = "SQSScheduledJobs"
    effect = "Allow"
    actions = [
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
    ]
    resources = [aws_sqs_queue.scheduled_jobs.arn]
  }
}

resource "aws_iam_policy" "backend_sqs" {
  name   = "${var.project_name}-backend-sqs"
  policy = data.aws_iam_policy_document.backend_sqs.json
}

resource "aws_iam_role_policy_attachment" "backend_sqs" {
  role       = aws_iam_role.backend.name
  policy_arn = aws_iam_policy.backend_sqs.arn
}

# EventBridge Scheduler management
data "aws_iam_policy_document" "backend_scheduler" {
  statement {
    sid    = "SchedulerManagement"
    effect = "Allow"
    actions = [
      "scheduler:CreateSchedule",
      "scheduler:DeleteSchedule",
      "scheduler:UpdateSchedule",
      "scheduler:GetSchedule",
    ]
    resources = ["arn:aws:scheduler:${var.aws_region}:${data.aws_caller_identity.current.account_id}:schedule/default/openbrowser-job-*"]
  }
  statement {
    sid     = "PassSchedulerRole"
    effect  = "Allow"
    actions = ["iam:PassRole"]
    resources = [aws_iam_role.scheduler.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_policy" "backend_scheduler" {
  name   = "${var.project_name}-backend-scheduler"
  policy = data.aws_iam_policy_document.backend_scheduler.json
}

resource "aws_iam_role_policy_attachment" "backend_scheduler" {
  role       = aws_iam_role.backend.name
  policy_arn = aws_iam_policy.backend_scheduler.arn
}
