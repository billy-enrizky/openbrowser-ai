# Auth & Scheduling

output "kms_auth_key_arn" {
  description = "KMS CMK ARN for auth profile encryption"
  value       = aws_kms_key.auth_data.arn
}

output "sqs_scheduled_jobs_queue_url" {
  description = "SQS queue URL for scheduled job messages"
  value       = aws_sqs_queue.scheduled_jobs.url
}

output "sqs_scheduled_jobs_dlq_url" {
  description = "SQS DLQ URL for failed scheduled job messages"
  value       = aws_sqs_queue.scheduled_jobs_dlq.url
}

output "scheduler_role_arn" {
  description = "EventBridge Scheduler IAM role ARN"
  value       = aws_iam_role.scheduler.arn
}
