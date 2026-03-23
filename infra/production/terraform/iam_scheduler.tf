# IAM role for EventBridge Scheduler to send messages to SQS

data "aws_iam_policy_document" "scheduler_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  name               = "${var.project_name}-scheduler-role"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume_role.json
}

data "aws_iam_policy_document" "scheduler_sqs" {
  statement {
    sid     = "SendToSQS"
    actions = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.scheduled_jobs.arn]
  }
}

resource "aws_iam_role_policy" "scheduler_sqs" {
  name   = "scheduler-sqs"
  role   = aws_iam_role.scheduler.id
  policy = data.aws_iam_policy_document.scheduler_sqs.json
}
