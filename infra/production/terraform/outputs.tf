# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "aws_region" {
  description = "AWS region for all resources"
  value       = var.aws_region
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

output "cognito_domain_url" {
  description = "Cognito hosted UI base URL"
  value       = "https://${aws_cognito_user_pool_domain.main.domain}.auth.${var.aws_region}.amazoncognito.com"
}

output "cognito_issuer" {
  description = "Cognito issuer URL for JWT verification"
  value       = "https://cognito-idp.${var.aws_region}.amazonaws.com/${aws_cognito_user_pool.main.id}"
}

output "cognito_callback_urls" {
  description = "Effective callback URLs configured on the Cognito app client"
  value       = local.cognito_callback_urls
}

output "cognito_logout_urls" {
  description = "Effective logout URLs configured on the Cognito app client"
  value       = local.cognito_logout_urls
}

output "cognito_primary_callback_url" {
  description = "Primary callback URL for frontend builds"
  value       = local.cognito_callback_urls[0]
}

output "cognito_primary_logout_url" {
  description = "Primary logout URL for frontend builds"
  value       = local.cognito_logout_urls[0]
}

output "cognito_oauth_scopes_string" {
  description = "Cognito OAuth scopes as a space-delimited string"
  value       = join(" ", var.cognito_oauth_scopes)
}

output "dynamodb_table_name" {
  description = "DynamoDB table for session/chats data"
  value       = aws_dynamodb_table.main.name
}

output "postgres_endpoint" {
  description = "RDS PostgreSQL endpoint hostname"
  value       = aws_db_instance.postgres.address
}

output "postgres_port" {
  description = "RDS PostgreSQL port"
  value       = aws_db_instance.postgres.port
}

output "postgres_db_name" {
  description = "RDS PostgreSQL database name"
  value       = aws_db_instance.postgres.db_name
}

output "postgres_username" {
  description = "RDS PostgreSQL master username"
  value       = aws_db_instance.postgres.username
}

output "postgres_password" {
  description = "RDS PostgreSQL master password (sensitive)"
  value       = random_password.postgres.result
  sensitive   = true
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

output "backend_instance_id" {
  description = "EC2 instance ID for the backend (used by deploy scripts via SSM)"
  value       = aws_instance.backend.id
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
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

# -----------------------------------------------------------------------------
# Landing Page
# -----------------------------------------------------------------------------

output "landing_url" {
  description = "CloudFront distribution URL for the landing page"
  value       = "https://${aws_cloudfront_distribution.landing.domain_name}"
}

output "landing_s3_bucket" {
  description = "S3 bucket name for landing page static files"
  value       = aws_s3_bucket.landing.id
}

output "landing_cloudfront_distribution_id" {
  description = "CloudFront distribution ID for landing page (for cache invalidations)"
  value       = aws_cloudfront_distribution.landing.id
}

# -----------------------------------------------------------------------------
# Waitlist
# -----------------------------------------------------------------------------

output "waitlist_api_url" {
  description = "API Gateway URL for waitlist submissions"
  value       = aws_apigatewayv2_api.waitlist.api_endpoint
}

output "waitlist_dynamodb_table" {
  description = "DynamoDB table name for waitlist entries"
  value       = aws_dynamodb_table.waitlist.name
}
