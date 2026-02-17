# Copy to terraform.tfvars and fill in. Do not commit terraform.tfvars.

aws_region   = "ca-central-1"
project_name = "openbrowser"

# Backend container image (manually pushed to ECR; see README for build/push steps):
# - Empty = use Terraform-created ECR repo (529206289231.dkr.ecr.ca-central-1.amazonaws.com/openbrowser-backend)
# - Set to full URI to use another registry
backend_image     = ""
backend_image_tag = "latest"

# Optional: use an existing Secrets Manager secret for LLM API keys
# Leave empty to have Terraform create a placeholder secret
# secrets_manager_secret_name = ""

# Optional: require JWT (Cognito) on API routes. Set true when auth is implemented.
enable_api_auth = false

# Optional: custom domain for frontend (requires ACM cert in us-east-1 for CloudFront)
# frontend_domain_name    = "app.example.com"
# frontend_acm_certificate_arn = "arn:aws:acm:us-east-1:..."

# Optional: Cognito hosted UI domain prefix
# cognito_domain_prefix = "openbrowser"
