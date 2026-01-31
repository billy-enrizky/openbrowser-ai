# -----------------------------------------------------------------------------
# API Gateway HTTP API: public /health, JWT-protected proxy to ALB
# -----------------------------------------------------------------------------

resource "aws_apigatewayv2_api" "http" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"
  description   = "OpenBrowser backend API"
}

resource "aws_apigatewayv2_vpc_link" "backend" {
  name               = "${var.project_name}-vpc-link"
  subnet_ids         = aws_subnet.private[*].id
  security_group_ids = [aws_security_group.backend.id]
  tags               = { Name = "${var.project_name}-vpc-link" }
}

resource "aws_apigatewayv2_integration" "backend" {
  api_id           = aws_apigatewayv2_api.http.id
  integration_type = "HTTP_PROXY"
  integration_uri  = aws_lb_listener.http.arn
  integration_method = "ANY"
  connection_type  = "VPC_LINK"
  connection_id    = aws_apigatewayv2_vpc_link.backend.id
  payload_format_version = "1.0"
  timeout_milliseconds   = 30000
}

# Public route: health check (no auth)
resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.backend.id}"
}

# CORS preflight (no auth)
resource "aws_apigatewayv2_route" "health_options" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "OPTIONS /health"
  target    = "integrations/${aws_apigatewayv2_integration.backend.id}"
}

# JWT authorizer (Cognito) - created when enable_api_auth is true
resource "aws_apigatewayv2_authorizer" "jwt" {
  count = var.enable_api_auth ? 1 : 0

  api_id          = aws_apigatewayv2_api.http.id
  authorizer_type = "JWT"
  name            = "${var.project_name}-jwt"

  identity_sources = ["$request.header.Authorization"]

  jwt_configuration {
    audience = [aws_cognito_user_pool_client.app.id]
    issuer   = "https://cognito-idp.${var.aws_region}.amazonaws.com/${aws_cognito_user_pool.main.id}"
  }
}

# Root path (optional JWT)
resource "aws_apigatewayv2_route" "root" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "ANY /"
  target    = "integrations/${aws_apigatewayv2_integration.backend.id}"

  authorization_type = var.enable_api_auth ? "JWT" : "NONE"
  authorizer_id      = var.enable_api_auth ? aws_apigatewayv2_authorizer.jwt[0].id : null
}

# Catch-all (optional JWT)
resource "aws_apigatewayv2_route" "proxy" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "ANY /{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.backend.id}"

  authorization_type = var.enable_api_auth ? "JWT" : "NONE"
  authorizer_id      = var.enable_api_auth ? aws_apigatewayv2_authorizer.jwt[0].id : null
}

# Default stage
resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
}
