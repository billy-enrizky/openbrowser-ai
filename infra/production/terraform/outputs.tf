# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "api_base_url" {
  description = "API Gateway HTTP API base URL (use for NEXT_PUBLIC_API_URL and WebSocket)"
  value       = "${aws_apigatewayv2_api.http.api_endpoint}/"
}

output "api_ws_url" {
  description = "WebSocket URL (use for NEXT_PUBLIC_WS_URL)"
  value       = "wss://${replace(replace(aws_apigatewayv2_api.http.api_endpoint, "https://", ""), "http://", "")}/ws"
}

output "frontend_url" {
  description = "CloudFront distribution URL for the frontend"
  value       = "https://${aws_cloudfront_distribution.frontend.domain_name}"
}

output "frontend_domain_name" {
  description = "CloudFront domain (for CNAME if using custom domain)"
  value       = aws_cloudfront_distribution.frontend.domain_name
}

output "frontend_s3_bucket" {
  description = "S3 bucket name for frontend static files (upload build output here)"
  value       = aws_s3_bucket.frontend.id
}

output "cognito_user_pool_id" {
  description = "Cognito User Pool ID (for auth integration)"
  value       = aws_cognito_user_pool.main.id
}

output "cognito_app_client_id" {
  description = "Cognito App Client ID"
  value       = aws_cognito_user_pool_client.app.id
}

output "cognito_domain" {
  description = "Cognito hosted UI domain prefix"
  value       = aws_cognito_user_pool_domain.main.domain
}

output "dynamodb_table_name" {
  description = "DynamoDB table for session/chats data"
  value       = aws_dynamodb_table.main.name
}

output "backend_secret_name" {
  description = "Secrets Manager secret name for backend API keys (if created)"
  value       = length(aws_secretsmanager_secret.backend_keys) > 0 ? aws_secretsmanager_secret.backend_keys[0].name : null
}

output "backend_ecr_repository_url" {
  description = "ECR repository URL for the backend image. Push your image here, then EC2 will pull it."
  value       = aws_ecr_repository.backend.repository_url
}

output "backend_image_uri" {
  description = "Full image URI the EC2 instance will pull (ECR repo:tag or backend_image variable)."
  value       = local.backend_image_uri
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID (for cache invalidations)"
  value       = aws_cloudfront_distribution.frontend.id
}
output "api_gateway_url" {
  description = "API Gateway endpoint URL"
  value       = aws_apigatewayv2_api.main.api_endpoint
}

output "api_gateway_id" {
  description = "API Gateway ID"
  value       = aws_apigatewayv2_api.main.id
}

output "alb_dns_name" {
  description = "Application Load Balancer DNS name"
  value       = aws_lb.main.dns_name
}

output "alb_arn" {
  description = "Application Load Balancer ARN"
  value       = aws_lb.main.arn
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "security_group_ec2_id" {
  description = "EC2 security group ID"
  value       = aws_security_group.ec2.id
}

output "security_group_alb_id" {
  description = "ALB security group ID"
  value       = aws_security_group.alb.id
}

output "ssm_google_api_key_name" {
  description = "SSM parameter name for Google API key"
  value       = aws_ssm_parameter.google_api_key.name
}

output "ssm_openai_api_key_name" {
  description = "SSM parameter name for OpenAI API key"
  value       = aws_ssm_parameter.openai_api_key.name
}

output "ssm_anthropic_api_key_name" {
  description = "SSM parameter name for Anthropic API key"
  value       = aws_ssm_parameter.anthropic_api_key.name
}

output "autoscaling_group_name" {
  description = "Auto Scaling Group name"
  value       = aws_autoscaling_group.app.name
}
