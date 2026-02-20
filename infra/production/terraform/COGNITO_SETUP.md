# Cognito SPA Setup Notes

This application uses Cognito Hosted UI with redirect-based PKCE flow.
That means callback/logout URLs are required.

## Defaults provided by Terraform

If you do not set overrides, Terraform configures:
- Callback URLs:
  - `https://<frontend-domain>/auth/callback`
  - `https://<frontend-domain>/auth/callback/`
  - `http://localhost:3000/auth/callback`
  - `http://localhost:3000/auth/callback/`
- Logout URLs:
  - `https://<frontend-domain>/login`
  - `https://<frontend-domain>/login/`
  - `http://localhost:3000/login`
  - `http://localhost:3000/login/`

Both with and without trailing slash are registered because Next.js is
configured with `trailingSlash: true`. The non-trailing-slash variant is
listed first and used as the primary `redirect_uri` in frontend builds.

`<frontend-domain>` is:
- `frontend_domain_name` (if set), or
- CloudFront distribution domain (e.g. `d3p903fxpmjf8v.cloudfront.net`).

## Override URLs explicitly (optional)

Use this in `terraform.tfvars` if you want strict custom URLs:

```hcl
cognito_callback_urls = [
  "https://app.example.com/auth/callback",
]

cognito_logout_urls = [
  "https://app.example.com/login",
]
```

## Required Cognito app client capabilities

Terraform configures the app client for SPA PKCE:
- `allowed_oauth_flows_user_pool_client = true`
- `allowed_oauth_flows = ["code"]`
- `allowed_oauth_scopes = ["openid", "email", "profile"]`
- `generate_secret = false`

## Frontend env must match

At build time (handled automatically by `deploy-frontend.sh`):
- `NEXT_PUBLIC_COGNITO_DOMAIN` must equal `cognito_domain_url` output.
- `NEXT_PUBLIC_COGNITO_CLIENT_ID` must equal `cognito_app_client_id`.
- `NEXT_PUBLIC_COGNITO_REDIRECT_URI` and `NEXT_PUBLIC_COGNITO_LOGOUT_URI` must be present in Cognito app client config.

## Current production values

These values are read from Terraform outputs (never hardcode them):

```bash
terraform output -raw cognito_user_pool_id      # ca-central-1_uU8gXJMFW
terraform output -raw cognito_app_client_id      # 5ovv2tf4r12f6m0q3kbh83nj6f
terraform output -raw cognito_domain_url         # Cognito hosted UI domain
terraform output -raw cognito_callback_urls      # Registered callback URLs
terraform output -raw cognito_logout_urls        # Registered logout URLs
```

## Troubleshooting

- **`redirect_mismatch` error**: The `redirect_uri` sent by the frontend does not match any URL registered in Cognito. Verify trailing slash handling.
- **`invalid_scope` error**: App client missing `openid`, `email`, or `profile` scope.
- **Login works locally but not in production**: Ensure CloudFront domain URLs are registered in Cognito callback/logout URLs.
