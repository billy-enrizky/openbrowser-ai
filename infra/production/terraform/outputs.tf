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
