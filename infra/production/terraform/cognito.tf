# -----------------------------------------------------------------------------
# Cognito: User Pool + App Client + Domain (Hosted UI + PKCE)
# -----------------------------------------------------------------------------

resource "random_id" "cognito_suffix" {
  byte_length = 4
}

locals {
  cognito_domain_prefix = var.cognito_domain_prefix != "" ? var.cognito_domain_prefix : "${var.project_name}-${random_id.cognito_suffix.hex}"
  frontend_base_url     = var.frontend_domain_name != "" ? "https://${var.frontend_domain_name}" : "https://${aws_cloudfront_distribution.frontend.domain_name}"
  # Both with and without trailing slash are registered because Next.js is
  # configured with trailingSlash: true. The non-trailing-slash variant is
  # listed first and used as the primary redirect_uri in frontend builds.
  cognito_callback_urls = length(var.cognito_callback_urls) > 0 ? var.cognito_callback_urls : [
    "${local.frontend_base_url}/auth/callback",
    "${local.frontend_base_url}/auth/callback/",
    "http://localhost:3000/auth/callback",
    "http://localhost:3000/auth/callback/",
  ]
  cognito_logout_urls = length(var.cognito_logout_urls) > 0 ? var.cognito_logout_urls : [
    "${local.frontend_base_url}/login",
    "${local.frontend_base_url}/login/",
    "http://localhost:3000/login",
    "http://localhost:3000/login/",
  ]
}

resource "aws_cognito_user_pool" "main" {
  name = "${var.project_name}-user-pool"

  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]

  password_policy {
    minimum_length    = 8
    require_lowercase = true
    require_uppercase = true
    require_numbers   = true
    require_symbols   = false
  }

  tags = { Name = "${var.project_name}-user-pool" }
}

resource "aws_cognito_user_pool_client" "app" {
  name         = "${var.project_name}-app-client"
  user_pool_id = aws_cognito_user_pool.main.id

  generate_secret               = false
  prevent_user_existence_errors = "ENABLED"

  explicit_auth_flows = [
    "ALLOW_USER_SRP_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
    "ALLOW_USER_PASSWORD_AUTH"
  ]

  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = var.cognito_oauth_scopes
  callback_urls                        = local.cognito_callback_urls
  logout_urls                          = local.cognito_logout_urls

  supported_identity_providers = ["COGNITO"]
}

resource "aws_cognito_user_pool_domain" "main" {
  domain       = local.cognito_domain_prefix
  user_pool_id = aws_cognito_user_pool.main.id
}
