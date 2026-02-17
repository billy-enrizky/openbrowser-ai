# Production Authentication Runbook

This repository uses:
- Cognito Hosted UI + OAuth2 Authorization Code flow (PKCE) in the frontend
- JWT verification in the backend (`AUTH_ENABLED=true`)
- Optional additional JWT enforcement at API Gateway (`enable_api_auth`)

## 1. Terraform settings

In `terraform.tfvars`:

```hcl
enable_backend_auth = true
enable_api_auth     = false

cors_origins = [
  "https://app.example.com",
]

# Optional overrides (if omitted, Terraform auto-generates these)
# cognito_callback_urls = ["https://app.example.com/auth/callback/"]
# cognito_logout_urls   = ["https://app.example.com/login/"]
```

Apply:

```bash
cd infra/production/terraform
terraform init
terraform apply
```

Note: backend auth env is injected via EC2 `user_data`; auth-related changes can replace the backend instance.

## 2. Frontend build env

Before `npm run build` in `frontend/`, set:

```bash
export NEXT_PUBLIC_API_URL=$(terraform -chdir=infra/production/terraform output -raw api_base_url)
export NEXT_PUBLIC_WS_URL=$(terraform -chdir=infra/production/terraform output -raw api_ws_url)
export NEXT_PUBLIC_AUTH_ENABLED=true
export NEXT_PUBLIC_COGNITO_DOMAIN=$(terraform -chdir=infra/production/terraform output -raw cognito_domain_url)
export NEXT_PUBLIC_COGNITO_CLIENT_ID=$(terraform -chdir=infra/production/terraform output -raw cognito_app_client_id)
export NEXT_PUBLIC_COGNITO_REDIRECT_URI=https://app.example.com/auth/callback/
export NEXT_PUBLIC_COGNITO_LOGOUT_URI=https://app.example.com/login/
export NEXT_PUBLIC_COGNITO_SCOPES="openid email profile"
```

Then build + upload:

```bash
cd frontend
npm ci
npm run build
aws s3 sync out/ s3://$(terraform -chdir=../infra/production/terraform output -raw frontend_s3_bucket)/ --delete
aws cloudfront create-invalidation --distribution-id $(terraform -chdir=../infra/production/terraform output -raw cloudfront_distribution_id) --paths "/*"
```

## 3. Backend auth env injected by Terraform

`backend.tf` now injects:
- `AUTH_ENABLED`
- `COGNITO_REGION`
- `COGNITO_USER_POOL_ID`
- `COGNITO_APP_CLIENT_ID`
- `CORS_ORIGINS`

No extra manual env setup is needed on EC2 for Cognito verification.

## 4. WebSocket auth behavior

Frontend sends JWT as query param: `wss://.../ws/<id>?token=<jwt>`.

Backend validates this token.  
If `enable_api_auth=true`, API Gateway also validates it (header or query token).

## 5. Quick verification

1. Open `https://app.example.com/login`.
2. Sign in via Cognito Hosted UI.
3. Confirm redirect to `/auth/callback/` then `/`.
4. Verify backend requests return `200`:
   - `GET /api/v1/models`
5. Verify WebSocket connects and task starts without `401/403`.

## 6. Most common failures

- `invalid_scope`: Cognito app client missing one of `openid/email/profile`.
- Redirect loop to `/`: `NEXT_PUBLIC_AUTH_ENABLED` was not set to `true` at frontend build time.
- `401 Unauthorized` from backend: backend Cognito env mismatch (wrong client ID/user pool).
- WebSocket `403` at handshake: token missing from WS URL or API Gateway authorizer misconfigured.
