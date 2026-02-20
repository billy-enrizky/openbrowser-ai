# Production Authentication Runbook

This repository uses:
- Cognito Hosted UI + OAuth2 Authorization Code flow (PKCE) in the frontend
- JWT verification in the backend (`AUTH_ENABLED=true`)

All traffic flows through CloudFront, which routes API requests to the ALB.

## 1. Terraform settings

In `terraform.tfvars`:

```hcl
enable_backend_auth = true

# CORS origins are auto-derived from CloudFront domain.
# Only set cors_origins if you need additional origins beyond CloudFront + localhost:
# cors_origins = ["https://custom-domain.example.com"]
```

Apply:

```bash
cd infra/production/terraform
terraform init
terraform apply
```

Note: backend auth env is injected via EC2 `user_data`; auth-related changes can replace the backend instance.

## 2. Frontend build env

The deploy script handles this automatically:

```bash
bash infra/production/scripts/deploy-frontend.sh
```

The script reads all required values from Terraform outputs and writes `.env.production.local`:
- `NEXT_PUBLIC_API_URL` -- CloudFront URL (derived from `frontend_url`)
- `NEXT_PUBLIC_WS_URL` -- WebSocket URL via CloudFront
- `NEXT_PUBLIC_AUTH_ENABLED=true`
- `NEXT_PUBLIC_COGNITO_DOMAIN` -- From `cognito_domain_url` output
- `NEXT_PUBLIC_COGNITO_CLIENT_ID` -- From `cognito_app_client_id` output
- `NEXT_PUBLIC_COGNITO_REDIRECT_URI` -- CloudFront domain + `/auth/callback`
- `NEXT_PUBLIC_COGNITO_LOGOUT_URI` -- CloudFront domain + `/login`
- `NEXT_PUBLIC_COGNITO_SCOPES` -- `openid email profile`

If building manually:

```bash
export NEXT_PUBLIC_API_URL="https://$(terraform -chdir=infra/production/terraform output -raw frontend_url | sed 's|https://||')"
export NEXT_PUBLIC_WS_URL="wss://$(terraform -chdir=infra/production/terraform output -raw frontend_url | sed 's|https://||')"
export NEXT_PUBLIC_AUTH_ENABLED=true
export NEXT_PUBLIC_COGNITO_DOMAIN=$(terraform -chdir=infra/production/terraform output -raw cognito_domain_url)
export NEXT_PUBLIC_COGNITO_CLIENT_ID=$(terraform -chdir=infra/production/terraform output -raw cognito_app_client_id)
export NEXT_PUBLIC_COGNITO_REDIRECT_URI="https://$(terraform -chdir=infra/production/terraform output -raw frontend_url | sed 's|https://||')/auth/callback"
export NEXT_PUBLIC_COGNITO_LOGOUT_URI="https://$(terraform -chdir=infra/production/terraform output -raw frontend_url | sed 's|https://||')/login"
export NEXT_PUBLIC_COGNITO_SCOPES="openid email profile"
cd frontend && npm ci && npm run build
```

## 3. Backend auth env injected by Terraform

`backend.tf` user_data and the deploy script inject:
- `AUTH_ENABLED`
- `COGNITO_REGION`
- `COGNITO_USER_POOL_ID`
- `COGNITO_APP_CLIENT_ID`
- `CORS_ORIGINS` (auto-derived from CloudFront domain + localhost)

No extra manual env setup is needed on EC2 for Cognito verification.

## 4. WebSocket auth behavior

Frontend sends JWT as query param: `wss://<cloudfront-domain>/api/v1/vnc/ws?task_id=X&token=<jwt>`.

Backend validates this token in the VNC proxy endpoint.

For the task polling endpoint (`GET /api/v1/tasks/{id}/events`), JWT is sent in the `Authorization` header as usual.

## 5. Quick verification

1. Open the CloudFront URL (e.g. `https://d3p903fxpmjf8v.cloudfront.net/login`).
2. Sign in via Cognito Hosted UI.
3. Confirm redirect to `/auth/callback` then `/`.
4. Verify backend requests return `200`:
   - `GET /api/v1/chats` (requires auth)
   - `GET /health` (no auth required)
5. Start a task and verify VNC WebSocket connects without `401/403`.

## 6. Most common failures

- **`invalid_scope`**: Cognito app client missing one of `openid/email/profile`.
- **Redirect loop to `/`**: `NEXT_PUBLIC_AUTH_ENABLED` was not set to `true` at frontend build time, or CloudFront is serving a stale cached `index.html`.
- **`401 Unauthorized` from backend**: Backend Cognito env mismatch (wrong client ID or user pool ID). Check container env via SSM.
- **VNC WebSocket `403` at handshake**: Token missing from WebSocket URL query param or token expired.
- **CloudFront 403 on `/auth/callback/`**: CloudFront Function not rewriting directory-like URIs to `index.html`. Verify the function is attached to the default cache behavior.
- **Stale frontend after deploy**: CloudFront cache not invalidated. Run `aws cloudfront create-invalidation --distribution-id <id> --paths "/*"`.
