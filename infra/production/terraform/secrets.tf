# -----------------------------------------------------------------------------
# Secrets Manager: placeholder for LLM API keys (optional)
# Create the secret manually or via CI, then set secrets_manager_secret_name.
# -----------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "backend_keys" {
  count = var.secrets_manager_secret_name == "" ? 1 : 0

  name        = "${var.project_name}/backend-api-keys"
  description = "LLM API keys for OpenBrowser backend (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.)"
  tags        = { Name = "${var.project_name}-backend-keys" }
}

# Placeholder value; replace with real keys via console or CLI
resource "aws_secretsmanager_secret_version" "backend_keys" {
  count = var.secrets_manager_secret_name == "" ? 1 : 0

  secret_id     = aws_secretsmanager_secret.backend_keys[0].id
  secret_string = jsonencode({
    OPENAI_API_KEY    = "replace-me"
    ANTHROPIC_API_KEY = "replace-me"
    GOOGLE_API_KEY    = "replace-me"
  })
}
