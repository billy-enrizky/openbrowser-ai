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

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID (for cache invalidations)"
  value       = aws_cloudfront_distribution.frontend.id
}
