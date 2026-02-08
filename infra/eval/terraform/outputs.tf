output "launch_template_id" {
  description = "Eval EC2 launch template ID -- use to launch spot instances"
  value       = aws_launch_template.eval.id
}

output "datasets_bucket" {
  description = "S3 bucket for evaluation datasets"
  value       = aws_s3_bucket.datasets.id
}

output "results_bucket" {
  description = "S3 bucket for evaluation results"
  value       = aws_s3_bucket.results.id
}

output "vpc_id" {
  description = "Eval VPC ID"
  value       = aws_vpc.eval.id
}

output "security_group_id" {
  description = "Eval security group ID"
  value       = aws_security_group.eval.id
}

output "ssm_google_api_key_name" {
  description = "SSM parameter name for Google API key"
  value       = aws_ssm_parameter.google_api_key.name
}

output "ssm_openai_api_key_name" {
  description = "SSM parameter name for OpenAI API key"
  value       = aws_ssm_parameter.openai_api_key.name
}

