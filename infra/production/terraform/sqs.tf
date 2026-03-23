# SQS queue for scheduled job execution messages

resource "aws_sqs_queue" "scheduled_jobs_dlq" {
  name                      = "${var.project_name}-scheduled-jobs-dlq"
  message_retention_seconds = 604800  # 7 days

  tags = { Name = "${var.project_name}-scheduled-jobs-dlq" }
}

resource "aws_sqs_queue" "scheduled_jobs" {
  name                       = "${var.project_name}-scheduled-jobs"
  visibility_timeout_seconds = 900     # 15 min (job execution timeout)
  message_retention_seconds  = 86400   # 24 hours
  receive_wait_time_seconds  = 20      # long polling

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.scheduled_jobs_dlq.arn
    maxReceiveCount     = 3
  })

  tags = { Name = "${var.project_name}-scheduled-jobs" }
}
